import math
import numpy as np
import torch
import torch.nn as nn

from .indexing import GAUSSIAN_VALUE_THRESHOLD, GaussianSpatialIndexer
from .renderer import HAS_CUDA_RENDERER, initialize_cuda_renderer, render_gaussians_cuda, gaussian_render_derivatives_cuda
from .utils import _has_jacobian


class GaussianVelocityField_Tiled(nn.Module):

    def __init__(self, num_gaussians, H, W, tile_size,
                 enable_jacobian=True, device='cuda', jacobian_mode='full',
                 init_scale_h=7.0, init_scale_w=7.0, render_tile_batch_size=32,
                 log_scale_min=0.0, cuda_renderer_policy='error'):
        super().__init__()
        self.num_gaussians   = num_gaussians
        self.H, self.W       = H, W
        self.tile_size       = tile_size
        self.enable_jacobian = enable_jacobian
        self.device          = device
        self.jacobian_mode   = jacobian_mode
        self.init_scale_h    = float(init_scale_h)
        self.init_scale_w    = float(init_scale_w)
        self.log_scale_min  = float(log_scale_min)
        self.log_scale_max  = float(np.log(H / 6))
        if not math.isfinite(self.log_scale_min) or self.log_scale_min < 0.0:
            raise ValueError("log_scale_min must be finite and >= log(1.0)")
        if self.log_scale_min > self.log_scale_max:
            raise ValueError("Minimum scale must not exceed H / 6")
        if not isinstance(render_tile_batch_size, int) or render_tile_batch_size < 0:
            raise ValueError("render_tile_batch_size must be a non-negative integer")
        self.render_tile_batch_size = render_tile_batch_size

        if self.jacobian_mode not in ("full", "trace_free"):
            raise ValueError(
                f"Unknown jacobian_mode={self.jacobian_mode!r}; "
                "expected 'full' or 'trace_free'."
            )

        if self.init_scale_h <= 1.0 or self.init_scale_w <= 1.0:
            raise ValueError("init_scale_h and init_scale_w must be > 1.0")

        self.cuda_renderer_policy = cuda_renderer_policy
        self.render_backend = initialize_cuda_renderer(device, cuda_renderer_policy)
        self._renderer_device_type = torch.device(device).type

        # Sample each coordinate against its own spatial extent.  Using H for
        # both axes leaves the right side of non-square (W > H) inputs without
        # initial Gaussian centers.
        mu_unit = torch.rand(num_gaussians, 2, device=device)
        mu_extent = mu_unit.new_tensor([H + 32, W + 32])
        self.mu = nn.Parameter(mu_unit * mu_extent - 16)

        # Sample actual axis scales independently, then store them in log space.
        # Equal bounds do not tie the two axes or force circular Gaussians.
        self.log_scale = nn.Parameter(
            torch.log(torch.stack([
                1.0 + torch.rand(num_gaussians, device=device) * (self.init_scale_h - 1.0),
                1.0 + torch.rand(num_gaussians, device=device) * (self.init_scale_w - 1.0)
            ], dim=1)).clamp(min=self.log_scale_min, max=self.log_scale_max))
        self.opacity_logit  = nn.Parameter(torch.zeros(num_gaussians, 1, device=device))
        self.rotation       = nn.Parameter(torch.rand(num_gaussians, 1, device=device) * torch.pi)
        self.local_velocity = nn.Parameter(torch.randn(num_gaussians, 2, device=device))

        if enable_jacobian:
            self.jacobian_dim = 3 if self.jacobian_mode == "trace_free" else 4
            self.J_params = nn.Parameter(
                0.05 * torch.randn(num_gaussians, self.jacobian_dim, device=device)
            )
        else:
            self.jacobian_dim = 0
            self.register_parameter("J_params", None)

        # tile 内像素偏移网格（固定 buffer，不参与梯度）
        dh_off = torch.arange(tile_size, dtype=torch.float32, device=device)
        dw_off = torch.arange(tile_size, dtype=torch.float32, device=device)
        lh, lw = torch.meshgrid(dh_off, dw_off, indexing='ij')
        self.register_buffer(
            '_local_coords',
            torch.stack([lh, lw], dim=-1).reshape(1, tile_size * tile_size, 2)
        )
        self._indexer_cache = None

    # ------------------------------------------------------------------
    def clamped_log_scale(self):
        """The same effective log-scales for rendering, indexing and density control."""
        return self.log_scale.clamp(min=self.log_scale_min, max=self.log_scale_max)

    def get_scales(self):
        return torch.exp(self.clamped_log_scale())

    @torch.no_grad()
    def clamp_scales_(self):
        """Keep stored parameters inside the bounds used by all evaluation paths."""
        self.log_scale.clamp_(min=self.log_scale_min, max=self.log_scale_max)

    def rebuild_indexer(self):
        with torch.no_grad():
            self._indexer_cache = GaussianSpatialIndexer(
                self.mu.detach().clone(),
                self.get_scales().detach().clone(),
                self.rotation.detach().clone(),
                (self.H, self.W),
                self.tile_size
            )

    def compute_covariance_inv_and_R(self):
        s     = self.get_scales()
        r     = self.rotation.squeeze(-1)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)

        R00, R01 =  cos_r, -sin_r
        R10, R11 =  sin_r,  cos_r

        inv_sh2 = 1.0 / (s[:, 0] ** 2 + 1e-6)
        inv_sw2 = 1.0 / (s[:, 1] ** 2 + 1e-6)

        Si_hh = R00 * R00 * inv_sh2 + R01 * R01 * inv_sw2
        Si_hw = R00 * R10 * inv_sh2 + R01 * R11 * inv_sw2
        Si_wh = Si_hw
        Si_ww = R10 * R10 * inv_sh2 + R11 * R11 * inv_sw2

        Sigma_inv = torch.stack([
            torch.stack([Si_hh, Si_hw], dim=-1),
            torch.stack([Si_wh, Si_ww], dim=-1)
        ], dim=-2)   # [N, 2, 2]

        R = torch.stack([
            torch.stack([R00, R01], dim=-1),
            torch.stack([R10, R11], dim=-1)
        ], dim=-2)   # [N, 2, 2]

        return Sigma_inv, R, s

    def build_jacobian_matrix(self, R):
        if not _has_jacobian(self):
            return None

        if self.jacobian_mode == "trace_free":
            a, b, c = self.J_params.unbind(-1)
            J_local = torch.stack([
                torch.stack([a,  b], dim=-1),
                torch.stack([c, -a], dim=-1)
            ], dim=-2)
        elif self.jacobian_mode == "full":
            a, b, c, d = self.J_params.unbind(-1)
            J_local = torch.stack([
                torch.stack([a, b], dim=-1),
                torch.stack([c, d], dim=-1)
            ], dim=-2)
        else:
            raise ValueError(f"Unknown jacobian_mode={self.jacobian_mode!r}")

        return torch.bmm(R, torch.bmm(J_local, R.transpose(1, 2)))

    # ------------------------------------------------------------------
    def _uses_cuda_renderer(self):
        # Support moving a model between CPU and CUDA after construction.
        device = self.mu.device
        if self._renderer_device_type != device.type:
            self.render_backend = initialize_cuda_renderer(device, self.cuda_renderer_policy)
            self._renderer_device_type = device.type
        return device.type == 'cuda' and self.render_backend == 'cuda'

    def forward(self, rebuild_index=False, compute_derivatives=False):
        """
        返回:
            flow       : [2, H, W]，通道0=vel_h（竖直），通道1=vel_w（水平）
            weight_map : [1, H, W]
            div        : [H, W] ∂vel_h/∂h + ∂vel_w/∂w，仅 compute_derivatives=True 时非 None
            vorticity  : [H, W] ∂vel_w/∂h − ∂vel_h/∂w，仅 compute_derivatives=True 时非 None
        """
        device = self.mu.device

        if rebuild_index or self._indexer_cache is None:
            self.rebuild_indexer()
        indexer = self._indexer_cache

        Sigma_inv, R, scales = self.compute_covariance_inv_and_R()
        opacity    = torch.sigmoid(self.opacity_logit)   # [N, 1]
        vel_global = torch.matmul(R, self.local_velocity.unsqueeze(-1)).squeeze(-1)  # [N, 2]

        J = self.build_jacobian_matrix(R)

        tile_counts     = indexer.cell_ptr[1:] - indexer.cell_ptr[:-1]
        active_tile_ids = torch.where(tile_counts > 0)[0]
        T = active_tile_ids.shape[0]
        _zf = torch.zeros(2, self.H, self.W, device=device)
        _zw = torch.zeros(1, self.H, self.W, device=device)
        if T == 0:
            zeros_hw = torch.zeros(self.H, self.W, device=device)
            return (_zf, _zw, zeros_hw, zeros_hw) if compute_derivatives else (_zf, _zw, None, None)

        if not compute_derivatives and self._uses_cuda_renderer():
            J_render = J if J is not None else torch.zeros(
                self.num_gaussians, 2, 2, device=device, dtype=self.mu.dtype
            )
            num_h_map, num_w_map, den_map = render_gaussians_cuda(
                self.mu,
                Sigma_inv,
                opacity,
                vel_global,
                J_render,
                indexer.cell_ptr,
                indexer.cell_indices,
                self.H,
                self.W,
                self.tile_size,
                indexer.grid_w,
            )
            weight_map = den_map.unsqueeze(0)
            W_safe = weight_map + 1e-8
            flow = torch.stack([num_h_map, num_w_map], dim=0) / W_safe
            return flow, weight_map, None, None

        K_max = int(tile_counts[active_tile_ids].max().item())

        starts = indexer.cell_ptr[active_tile_ids]
        counts = tile_counts[active_tile_ids]
        total  = int(counts.sum().item())

        tile_idx  = torch.repeat_interleave(torch.arange(T, device=device), counts)
        seg_start = starts.repeat_interleave(counts)
        seg_iota  = torch.arange(total, device=device) - \
                    torch.repeat_interleave(
                        torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                   counts.cumsum(0)[:-1]]),
                        counts)
        all_gids  = indexer.cell_indices[seg_start + seg_iota]

        padded_gids = torch.zeros(T, K_max, dtype=torch.long, device=device)
        valid_gmask = torch.zeros(T, K_max, dtype=torch.bool,  device=device)
        padded_gids[tile_idx, seg_iota] = all_gids
        valid_gmask[tile_idx, seg_iota] = True

        flat_ids    = padded_gids.reshape(-1)
        cur_mu      = self.mu[flat_ids].reshape(T, K_max, 2)
        cur_si      = Sigma_inv[flat_ids].reshape(T, K_max, 2, 2)
        cur_opacity = opacity[flat_ids].reshape(T, K_max, 1)
        cur_vel     = vel_global[flat_ids].reshape(T, K_max, 2)
        cur_J       = J[flat_ids].reshape(T, K_max, 2, 2) if J is not None else None

        tile_rows    = active_tile_ids // indexer.grid_w
        tile_cols    = active_tile_ids  % indexer.grid_w
        tile_origins = torch.stack([
            (tile_rows * self.tile_size).float(),
            (tile_cols * self.tile_size).float()
        ], dim=-1).unsqueeze(1)   # [T, 1, 2]

        pixel_coords = tile_origins + self._local_coords   # [T, P, 2]

        pixel_valid = (
            (pixel_coords[..., 0] < self.H) &
            (pixel_coords[..., 1] < self.W)
        )
        pixel_coords = torch.where(
            pixel_valid.unsqueeze(-1),
            pixel_coords,
            tile_origins.expand_as(pixel_coords)
        )

        delta = pixel_coords.unsqueeze(2) - cur_mu.unsqueeze(1)   # [T, P, K, 2]

        dh    = delta[..., 0]
        dw    = delta[..., 1]
        Si_hh = cur_si[..., 0, 0]
        Si_hw = cur_si[..., 0, 1]
        Si_ww = cur_si[..., 1, 1]

        term = (dh * dh * Si_hh.unsqueeze(1)
              + 2 * dh * dw * Si_hw.unsqueeze(1)
              + dw * dw * Si_ww.unsqueeze(1))   # [T, P, K]

        valid_weight_mask = valid_gmask.unsqueeze(1) & pixel_valid.unsqueeze(2)
        gauss_w = torch.exp(-0.5 * term)
        valid_weight_mask = valid_weight_mask & (gauss_w >= GAUSSIAN_VALUE_THRESHOLD)
        gauss_w = torch.where(valid_weight_mask, gauss_w, 0.0)
        weights = gauss_w * cur_opacity.squeeze(-1).unsqueeze(1)

        if cur_J is not None:
            vel_h_aff = cur_J[..., 0, 0].unsqueeze(1) * dh + cur_J[..., 0, 1].unsqueeze(1) * dw
            vel_w_aff = cur_J[..., 1, 0].unsqueeze(1) * dh + cur_J[..., 1, 1].unsqueeze(1) * dw
            vel_aff   = torch.stack([vel_h_aff, vel_w_aff], dim=-1)
        else:
            vel_aff = 0.0

        vel_all      = cur_vel.unsqueeze(1) + vel_aff   # [T, P, K, 2]
        w3           = weights.unsqueeze(-1)
        weighted_vel = (w3 * vel_all).sum(dim=2)        # [T, P, 2]
        weight_sum   = weights.sum(dim=2, keepdim=True) # [T, P, 1]

        global_h   = pixel_coords[..., 0].long()
        global_w   = pixel_coords[..., 1].long()
        global_idx = global_h * self.W + global_w       # [T, P]

        valid_flat = pixel_valid.reshape(-1)
        gidx_flat  = global_idx.reshape(-1)[valid_flat]

        # ── 解析散度 / 旋度路径 ────────────────────────────────────────────
        if compute_derivatives:
            neg_Si_dh = -(Si_hh.unsqueeze(1) * dh + Si_hw.unsqueeze(1) * dw)
            neg_Si_dw = -(Si_hw.unsqueeze(1) * dh + Si_ww.unsqueeze(1) * dw)
            dlogw_d = torch.stack([neg_Si_dh, neg_Si_dw], dim=-1)   # [T, P, K, 2]

            J_contrib = cur_J if cur_J is not None else \
                        torch.zeros(T, K_max, 2, 2, device=device)

            outer  = vel_all.unsqueeze(-1) * dlogw_d.unsqueeze(-2)
            dN_d   = w3.unsqueeze(-1) * (J_contrib.unsqueeze(1) + outer)   # [T, P, K, 2, 2]
            dW_d   = w3.squeeze(-1).unsqueeze(-1) * dlogw_d

            dN_sum = dN_d.sum(dim=2)   # [T, P, 2, 2]
            dW_sum = dW_d.sum(dim=2)   # [T, P, 2]

            deriv_src = torch.cat([
                dN_sum[..., 0, 0:1],   # ∂N_vel_h/∂h
                dN_sum[..., 0, 1:2],   # ∂N_vel_h/∂w
                dN_sum[..., 1, 0:1],   # ∂N_vel_w/∂h
                dN_sum[..., 1, 1:2],   # ∂N_vel_w/∂w
                dW_sum[..., 0:1],      # ∂W/∂h
                dW_sum[..., 1:2],      # ∂W/∂w
            ], dim=-1)   # [T, P, 6]

            out9 = torch.zeros(9, self.H * self.W, device=device)
            src9 = torch.cat([
                weighted_vel.reshape(-1, 2)[valid_flat].T,
                weight_sum.reshape(-1, 1)[valid_flat].T,
                deriv_src.reshape(-1, 6)[valid_flat].T,
            ], dim=0)
            out9.scatter_add_(1, gidx_flat.unsqueeze(0).expand(9, -1), src9)

            vel_h_map  = out9[0:1].reshape(1, self.H, self.W)
            vel_w_map  = out9[1:2].reshape(1, self.H, self.W)
            weight_map = out9[2:3].reshape(1, self.H, self.W)
            dN_hh_map  = out9[3].reshape(self.H, self.W)
            dN_hw_map  = out9[4].reshape(self.H, self.W)
            dN_wh_map  = out9[5].reshape(self.H, self.W)
            dN_ww_map  = out9[6].reshape(self.H, self.W)
            dW_h_map   = out9[7].reshape(self.H, self.W)
            dW_w_map   = out9[8].reshape(self.H, self.W)

            W_safe = weight_map + 1e-8
            U_h    = vel_h_map / W_safe
            U_w    = vel_w_map / W_safe
            flow   = torch.cat([U_h, U_w], dim=0)   # [2,H,W]

            dU_h_dh = (dN_hh_map - U_h.squeeze(0) * dW_h_map) / W_safe.squeeze(0)
            dU_h_dw = (dN_hw_map - U_h.squeeze(0) * dW_w_map) / W_safe.squeeze(0)
            dU_w_dh = (dN_wh_map - U_w.squeeze(0) * dW_h_map) / W_safe.squeeze(0)
            dU_w_dw = (dN_ww_map - U_w.squeeze(0) * dW_w_map) / W_safe.squeeze(0)

            div       = dU_h_dh + dU_w_dw
            vorticity = dU_w_dh - dU_h_dw

            return flow, weight_map, div, vorticity

        out3 = torch.zeros(3, self.H * self.W, device=device)
        src3 = torch.cat([
            weighted_vel.reshape(-1, 2)[valid_flat].T,
            weight_sum.reshape(-1, 1)[valid_flat].T
        ], dim=0)
        out3.scatter_add_(1, gidx_flat.unsqueeze(0).expand(3, -1), src3)

        weight_map = out3[2:].reshape(1, self.H, self.W)
        W_safe     = weight_map + 1e-31
        U_h = out3[0:1].reshape(1, self.H, self.W) / W_safe
        U_w = out3[1:2].reshape(1, self.H, self.W) / W_safe
        flow = torch.cat([U_h, U_w], dim=0)

        return flow, weight_map, None, None

    # ------------------------------------------------------------------
    def _render_native_batched(self, tile_batch_size, compute_derivatives=True):
        """Render native pixels in bounded tile batches using scalar derivative sums."""
        if self._indexer_cache is None:
            self.rebuild_indexer()
        indexer = self._indexer_cache
        device = self.mu.device
        H, W = self.H, self.W
        si, rotation, _ = self.compute_covariance_inv_and_R()
        opacity = self.opacity_logit.sigmoid().squeeze(-1)
        velocity = (rotation @ self.local_velocity.unsqueeze(-1)).squeeze(-1)
        jacobian = self.build_jacobian_matrix(rotation)
        counts_all = indexer.cell_ptr[1:] - indexer.cell_ptr[:-1]
        active = torch.where(counts_all > 0)[0]
        channels = 9 if compute_derivatives else 3
        accum = torch.zeros(channels, H * W, device=device)

        for tile_ids in active.split(tile_batch_size):
            if tile_ids.numel() == 0:
                continue
            counts = counts_all[tile_ids]
            k_max = int(counts.max().item())
            offset = torch.arange(k_max, device=device).unsqueeze(0)
            gaussian_valid = offset < counts.unsqueeze(1)
            positions = indexer.cell_ptr[tile_ids].unsqueeze(1) + offset
            safe_positions = torch.where(gaussian_valid, positions, 0)
            gids = indexer.cell_indices[safe_positions]
            mu = self.mu[gids]
            cur_si = si[gids]
            cur_vel = velocity[gids]
            cur_j = jacobian[gids] if jacobian is not None else None
            origins = torch.stack([
                (tile_ids // indexer.grid_w * self.tile_size).float(),
                (tile_ids % indexer.grid_w * self.tile_size).float(),
            ], dim=-1).unsqueeze(1)
            coords = origins + self._local_coords
            pixel_valid = (coords[..., 0] < H) & (coords[..., 1] < W)
            coords = torch.where(pixel_valid.unsqueeze(-1), coords, origins.expand_as(coords))
            delta = coords.unsqueeze(2) - mu.unsqueeze(1)
            dh, dw = delta[..., 0], delta[..., 1]
            si_hh = cur_si[..., 0, 0].unsqueeze(1)
            si_hw = cur_si[..., 0, 1].unsqueeze(1)
            si_ww = cur_si[..., 1, 1].unsqueeze(1)
            term = dh * dh * si_hh + 2 * dh * dw * si_hw + dw * dw * si_ww
            gaussian = torch.exp(-0.5 * term)
            valid = gaussian_valid.unsqueeze(1) & pixel_valid.unsqueeze(2)
            valid = valid & (gaussian >= GAUSSIAN_VALUE_THRESHOLD)
            weights = torch.where(valid, gaussian, 0.0) * opacity[gids].unsqueeze(1)
            if cur_j is None:
                vel_affine = 0.0
            else:
                vel_affine = torch.stack([
                    cur_j[..., 0, 0].unsqueeze(1) * dh + cur_j[..., 0, 1].unsqueeze(1) * dw,
                    cur_j[..., 1, 0].unsqueeze(1) * dh + cur_j[..., 1, 1].unsqueeze(1) * dw,
                ], dim=-1)
            vel_all = cur_vel.unsqueeze(1) + vel_affine
            numerator = (weights.unsqueeze(-1) * vel_all).sum(dim=2)
            denominator = weights.sum(dim=2, keepdim=True)
            sources = [numerator, denominator]
            if compute_derivatives:
                dlog_h = -(si_hh * dh + si_hw * dw)
                dlog_w = -(si_hw * dh + si_ww * dw)
                j = cur_j if cur_j is not None else torch.zeros(
                    len(tile_ids), k_max, 2, 2, device=device
                )
                vh, vw = vel_all[..., 0], vel_all[..., 1]
                dN_hh = (weights * (j[..., 0, 0].unsqueeze(1) + vh * dlog_h)).sum(dim=2)
                dN_hw = (weights * (j[..., 0, 1].unsqueeze(1) + vh * dlog_w)).sum(dim=2)
                dN_wh = (weights * (j[..., 1, 0].unsqueeze(1) + vw * dlog_h)).sum(dim=2)
                dN_ww = (weights * (j[..., 1, 1].unsqueeze(1) + vw * dlog_w)).sum(dim=2)
                dW_h = (weights * dlog_h).sum(dim=2)
                dW_w = (weights * dlog_w).sum(dim=2)
                sources.append(torch.stack([dN_hh, dN_hw, dN_wh, dN_ww, dW_h, dW_w], dim=-1))
            source = torch.cat(sources, dim=-1)
            keep = pixel_valid.reshape(-1)
            flat_idx = (coords[..., 0].long() * W + coords[..., 1].long()).reshape(-1)[keep]
            accum.scatter_add_(
                1, flat_idx.unsqueeze(0).expand(channels, -1),
                source.reshape(-1, channels)[keep].T,
            )

        maps = accum.reshape(channels, H, W)
        weight_map = maps[2:3]
        safe_weight = weight_map + 1e-8
        flow = maps[:2] / safe_weight
        if not compute_derivatives:
            return flow, weight_map, None, None
        safe = safe_weight.squeeze(0)
        dUh_dh = (maps[3] - flow[0] * maps[7]) / safe
        dUh_dw = (maps[4] - flow[0] * maps[8]) / safe
        dUw_dh = (maps[5] - flow[1] * maps[7]) / safe
        dUw_dw = (maps[6] - flow[1] * maps[8]) / safe
        return flow, weight_map, dUh_dh + dUw_dw, dUw_dh - dUh_dw

    def _render_native_cuda(self, compute_derivatives=True):
        """Use the training CUDA tile renderer, adding analytic derivative sums."""
        if not compute_derivatives:
            return self.forward(compute_derivatives=False)
        indexer = self._indexer_cache
        si, rotation, _ = self.compute_covariance_inv_and_R()
        opacity = self.opacity_logit.sigmoid()
        velocity = (rotation @ self.local_velocity.unsqueeze(-1)).squeeze(-1)
        jacobian = self.build_jacobian_matrix(rotation)
        if jacobian is None:
            jacobian = self.mu.new_zeros(self.num_gaussians, 2, 2)
        maps = gaussian_render_derivatives_cuda(
            self.mu, si, opacity, velocity, jacobian,
            indexer.cell_ptr, indexer.cell_indices,
            self.H, self.W, self.tile_size, indexer.grid_w,
        )
        weight_map = maps[2:3]
        safe_weight = weight_map + 1e-8
        flow = maps[:2] / safe_weight
        safe = safe_weight.squeeze(0)
        dUh_dh = (maps[3] - flow[0] * maps[7]) / safe
        dUh_dw = (maps[4] - flow[0] * maps[8]) / safe
        dUw_dh = (maps[5] - flow[1] * maps[7]) / safe
        dUw_dw = (maps[6] - flow[1] * maps[8]) / safe
        return flow, weight_map, dUh_dh + dUw_dw, dUw_dh - dUh_dw

    def render_at_resolution(self, eval_H: int, eval_W: int,
                             compute_derivatives: bool = True):
        """
        在任意目标分辨率 (eval_H, eval_W) 上直接光栅化高斯场。

        **所有高斯参数保持不变**——高斯定义在训练坐标空间 [0,H)×[0,W) 中，
        eval 时只改变查询网格的密度：
            - 训练时每格查询整数坐标 0, 1, 2, ..., H-1
            - eval 时查询分数坐标 0, H/eval_H, 2H/eval_H, ..., H-H/eval_H
              即在同一坐标空间内插入更多采样点，天然得到更高分辨率的输出。

        Tile 索引器（Gaussian→tile 分配）基于 [0,H)×[0,W) 坐标，每次最终渲染前
        都按当前参数重建。每个 tile 内的查询点数从 ts×ts 变为 pts_h×pts_w
        （pts_h = ceil(ts * eval_H/H)），分数偏移量为 [ih * H/eval_H, iw * W/eval_W]。
        散射目标从 H×W 的 flat buffer 换成 eval_H×eval_W 的 flat buffer：
            output_i = round(gauss_h * eval_H/H)，output_j = round(gauss_w * eval_W/W)

        仅应在 torch.no_grad() + model.eval() 上下文中调用。

        原分辨率无梯度推理在 CUDA 上使用与训练共用的融合 tile 内核，
        直接累加解析导数；非 CUDA 后端默认分批处理 32 个 tile。
        render_tile_batch_size=0、其他输出分辨率或启用梯度时保留原路径。
        """
        # Training can leave a cache from an earlier optimizer step. Refresh
        # before dispatch so both batched and ordinary final rendering use
        # the final centers, scales and rotations.
        self.rebuild_indexer()
        tile_batch_size = getattr(self, 'render_tile_batch_size', 32)
        if (tile_batch_size > 0 and not torch.is_grad_enabled()
                and (eval_H, eval_W) == (self.H, self.W)):
            if self._uses_cuda_renderer():
                return self._render_native_cuda(compute_derivatives)
            return self._render_native_batched(tile_batch_size, compute_derivatives)

        device = self.mu.device
        sf_h   = eval_H / self.H   # 行方向缩放因子
        sf_w   = eval_W / self.W   # 列方向缩放因子

        # ── Gaussian 参数（完全不变）──────────────────────────────────────
        indexer = self._indexer_cache

        Sigma_inv, R, _ = self.compute_covariance_inv_and_R()
        opacity    = torch.sigmoid(self.opacity_logit)
        vel_global = torch.matmul(R, self.local_velocity.unsqueeze(-1)).squeeze(-1)

        J = self.build_jacobian_matrix(R)

        # ── Tile 结构（索引器不变，只改每 tile 内的查询点）────────────────
        tile_counts     = indexer.cell_ptr[1:] - indexer.cell_ptr[:-1]
        active_tile_ids = torch.where(tile_counts > 0)[0]
        T = active_tile_ids.shape[0]

        zf = torch.zeros(2, eval_H, eval_W, device=device)
        zw = torch.zeros(1, eval_H, eval_W, device=device)
        if T == 0:
            zh = torch.zeros(eval_H, eval_W, device=device)
            return (zf, zw, zh, zh) if compute_derivatives else (zf, zw, None, None)

        K_max = int(tile_counts[active_tile_ids].max().item())
        starts = indexer.cell_ptr[active_tile_ids]
        counts = tile_counts[active_tile_ids]
        total  = int(counts.sum().item())

        tile_idx  = torch.repeat_interleave(torch.arange(T, device=device), counts)
        seg_start = starts.repeat_interleave(counts)
        seg_iota  = torch.arange(total, device=device) - \
                    torch.repeat_interleave(
                        torch.cat([torch.zeros(1, dtype=torch.long, device=device),
                                   counts.cumsum(0)[:-1]]),
                        counts)
        all_gids = indexer.cell_indices[seg_start + seg_iota]

        padded_gids = torch.zeros(T, K_max, dtype=torch.long, device=device)
        valid_gmask = torch.zeros(T, K_max, dtype=torch.bool,  device=device)
        padded_gids[tile_idx, seg_iota] = all_gids
        valid_gmask[tile_idx, seg_iota] = True

        flat_ids    = padded_gids.reshape(-1)
        cur_mu      = self.mu[flat_ids].reshape(T, K_max, 2)
        cur_si      = Sigma_inv[flat_ids].reshape(T, K_max, 2, 2)
        cur_opacity = opacity[flat_ids].reshape(T, K_max, 1)
        cur_vel     = vel_global[flat_ids].reshape(T, K_max, 2)
        cur_J       = J[flat_ids].reshape(T, K_max, 2, 2) if J is not None else None

        tile_rows    = active_tile_ids // indexer.grid_w
        tile_cols    = active_tile_ids  % indexer.grid_w
        tile_origins = torch.stack([
            (tile_rows * self.tile_size).float(),
            (tile_cols * self.tile_size).float()
        ], dim=-1).unsqueeze(1)   # [T, 1, 2]，Gaussian 坐标空间

        # ── 分数查询网格（关键：密度提升，参数不动）────────────────────────
        # 每 tile 在行/列方向各覆盖 tile_size 个 Gaussian 坐标单位；
        # eval 分辨率下，每个 Gaussian 坐标单位对应 sf_h/sf_w 个输出像素，
        # 因此每 tile 需要 pts_h × pts_w 个查询点。
        ts    = self.tile_size
        pts_h = math.ceil(ts * sf_h)
        pts_w = math.ceil(ts * sf_w)

        # 查询点在 tile 内的 Gaussian 坐标偏移：ih / sf_h, iw / sf_w
        dh_off = torch.arange(pts_h, device=device, dtype=torch.float32) / sf_h
        dw_off = torch.arange(pts_w, device=device, dtype=torch.float32) / sf_w
        lh, lw = torch.meshgrid(dh_off, dw_off, indexing='ij')
        eval_local = torch.stack([lh, lw], dim=-1).reshape(1, pts_h * pts_w, 2)

        pixel_coords = tile_origins + eval_local   # [T, P, 2]，Gaussian 坐标

        # 有效性：仍在训练坐标空间 [0, H)×[0, W) 内
        pixel_valid = (
            (pixel_coords[..., 0] >= 0) &
            (pixel_coords[..., 0] < self.H) &
            (pixel_coords[..., 1] >= 0) &
            (pixel_coords[..., 1] < self.W)
        )
        pixel_coords = torch.where(
            pixel_valid.unsqueeze(-1),
            pixel_coords,
            tile_origins.expand_as(pixel_coords)
        )

        # ── Gaussian 响应（同 forward()）──────────────────────────────────
        delta = pixel_coords.unsqueeze(2) - cur_mu.unsqueeze(1)
        dh    = delta[..., 0];  dw    = delta[..., 1]
        Si_hh = cur_si[..., 0, 0];  Si_hw = cur_si[..., 0, 1];  Si_ww = cur_si[..., 1, 1]

        term    = (dh * dh * Si_hh.unsqueeze(1)
                 + 2 * dh * dw * Si_hw.unsqueeze(1)
                 + dw * dw * Si_ww.unsqueeze(1))
        valid_weight_mask = valid_gmask.unsqueeze(1) & pixel_valid.unsqueeze(2)
        gauss_w = torch.exp(-0.5 * term)
        valid_weight_mask = valid_weight_mask & (gauss_w >= GAUSSIAN_VALUE_THRESHOLD)
        gauss_w = torch.where(valid_weight_mask, gauss_w, 0.0)
        weights = gauss_w * cur_opacity.squeeze(-1).unsqueeze(1)

        if cur_J is not None:
            vel_aff = torch.stack([
                cur_J[..., 0, 0].unsqueeze(1) * dh + cur_J[..., 0, 1].unsqueeze(1) * dw,
                cur_J[..., 1, 0].unsqueeze(1) * dh + cur_J[..., 1, 1].unsqueeze(1) * dw,
            ], dim=-1)
        else:
            vel_aff = 0.0

        vel_all      = cur_vel.unsqueeze(1) + vel_aff
        w3           = weights.unsqueeze(-1)
        weighted_vel = (w3 * vel_all).sum(dim=2)
        weight_sum   = weights.sum(dim=2, keepdim=True)

        # ── 散射到 eval_H × eval_W buffer（核心变化）──────────────────────
        # Gaussian 坐标 → 输出像素索引：round(h * sf_h), round(w * sf_w)
        out_h      = (pixel_coords[..., 0] * sf_h).round().long().clamp(0, eval_H - 1)
        out_w      = (pixel_coords[..., 1] * sf_w).round().long().clamp(0, eval_W - 1)
        global_idx = out_h * eval_W + out_w   # [T, P]

        valid_flat = pixel_valid.reshape(-1)
        gidx_flat  = global_idx.reshape(-1)[valid_flat]

        if compute_derivatives:
            neg_Si_dh = -(Si_hh.unsqueeze(1) * dh + Si_hw.unsqueeze(1) * dw)
            neg_Si_dw = -(Si_hw.unsqueeze(1) * dh + Si_ww.unsqueeze(1) * dw)
            dlogw_d = torch.stack([neg_Si_dh, neg_Si_dw], dim=-1)

            J_contrib = cur_J if cur_J is not None else \
                        torch.zeros(T, K_max, 2, 2, device=device)
            outer  = vel_all.unsqueeze(-1) * dlogw_d.unsqueeze(-2)
            dN_d   = w3.unsqueeze(-1) * (J_contrib.unsqueeze(1) + outer)
            dW_d   = w3.squeeze(-1).unsqueeze(-1) * dlogw_d

            dN_sum = dN_d.sum(dim=2)
            dW_sum = dW_d.sum(dim=2)

            deriv_src = torch.cat([
                dN_sum[..., 0, 0:1], dN_sum[..., 0, 1:2],
                dN_sum[..., 1, 0:1], dN_sum[..., 1, 1:2],
                dW_sum[..., 0:1],    dW_sum[..., 1:2],
            ], dim=-1)

            out9 = torch.zeros(9, eval_H * eval_W, device=device)
            src9 = torch.cat([
                weighted_vel.reshape(-1, 2)[valid_flat].T,
                weight_sum.reshape(-1, 1)[valid_flat].T,
                deriv_src.reshape(-1, 6)[valid_flat].T,
            ], dim=0)
            out9.scatter_add_(1, gidx_flat.unsqueeze(0).expand(9, -1), src9)

            vel_h_map = out9[0:1].reshape(1, eval_H, eval_W)
            vel_w_map = out9[1:2].reshape(1, eval_H, eval_W)
            weight_map = out9[2:3].reshape(1, eval_H, eval_W)
            dN_hh_map = out9[3].reshape(eval_H, eval_W)
            dN_hw_map = out9[4].reshape(eval_H, eval_W)
            dN_wh_map = out9[5].reshape(eval_H, eval_W)
            dN_ww_map = out9[6].reshape(eval_H, eval_W)
            dW_h_map  = out9[7].reshape(eval_H, eval_W)
            dW_w_map  = out9[8].reshape(eval_H, eval_W)

            W_safe = weight_map + 1e-8
            U_h    = vel_h_map / W_safe
            U_w    = vel_w_map / W_safe
            flow   = torch.cat([U_h, U_w], dim=0)

            dU_h_dh = (dN_hh_map - U_h.squeeze(0) * dW_h_map) / W_safe.squeeze(0)
            dU_h_dw = (dN_hw_map - U_h.squeeze(0) * dW_w_map) / W_safe.squeeze(0)
            dU_w_dh = (dN_wh_map - U_w.squeeze(0) * dW_h_map) / W_safe.squeeze(0)
            dU_w_dw = (dN_ww_map - U_w.squeeze(0) * dW_w_map) / W_safe.squeeze(0)

            div       = dU_h_dh + dU_w_dw
            vorticity = dU_w_dh - dU_h_dw
            return flow, weight_map, div, vorticity

        out3 = torch.zeros(3, eval_H * eval_W, device=device)
        src3 = torch.cat([
            weighted_vel.reshape(-1, 2)[valid_flat].T,
            weight_sum.reshape(-1, 1)[valid_flat].T,
        ], dim=0)
        out3.scatter_add_(1, gidx_flat.unsqueeze(0).expand(3, -1), src3)

        weight_map = out3[2:].reshape(1, eval_H, eval_W)
        W_safe     = weight_map + 1e-8
        U_h = out3[0:1].reshape(1, eval_H, eval_W) / W_safe
        U_w = out3[1:2].reshape(1, eval_H, eval_W) / W_safe
        return torch.cat([U_h, U_w], dim=0), weight_map, None, None
