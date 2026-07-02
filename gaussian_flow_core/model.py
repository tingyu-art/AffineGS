import math
import numpy as np
import torch
import torch.nn as nn

from .indexing import GaussianSpatialIndexer
from .utils import _has_jacobian

try:
    from gaussian_render_ext.gaussian_render import gaussian_render_cuda
    HAS_GAUSSIAN_RENDER_CUDA = True
except Exception:
    gaussian_render_cuda = None
    HAS_GAUSSIAN_RENDER_CUDA = False


class GaussianVelocityField_Tiled(nn.Module):

    def __init__(self, num_gaussians, H, W, tile_size, events_map,
                 density_alpha, enable_jacobian=True, device='cuda'):
        super().__init__()
        self.num_gaussians   = num_gaussians
        self.H, self.W       = H, W
        self.tile_size       = tile_size
        self.enable_jacobian = enable_jacobian
        self.device          = device

        if events_map is not None:
            self.mu = nn.Parameter(
                self.initialize_gaussian_positions(events_map, H, W, num_gaussians, density_alpha))
        else:
            self.mu = nn.Parameter(torch.rand(num_gaussians, 2, device=device) * (H + 32) - 16)

        self.log_scale = nn.Parameter(
            torch.stack([
                torch.rand(num_gaussians, device=device) * np.log(10.0),
                torch.rand(num_gaussians, device=device) * np.log(7.0)
            ], dim=1))
        self.opacity_logit  = nn.Parameter(torch.zeros(num_gaussians, 1, device=device))
        self.rotation       = nn.Parameter(torch.rand(num_gaussians, 1, device=device) * torch.pi)
        self.local_velocity = nn.Parameter(torch.randn(num_gaussians, 2, device=device))

        if enable_jacobian:
            self.J_params = nn.Parameter(0.05 * torch.randn(num_gaussians, 4, device=device))
        else:
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
    def initialize_gaussian_positions(self, events_map, H, W, num_gaussians, alpha):
        device = events_map.device
        with torch.no_grad():
            events_map      = torch.log1p(events_map)
            events_map_norm = events_map / (events_map.sum() + 1e-8)
            uniform         = torch.ones_like(events_map) / (H * W)
            density         = alpha * events_map_norm + (1 - alpha) * uniform
            flat            = density.reshape(-1)
            sampled_idx     = torch.multinomial(flat, num_gaussians, replacement=True)
            mu_h = (sampled_idx // W).float() + torch.randn(num_gaussians, device=device) * 0.5
            mu_w = (sampled_idx  % W).float() + torch.randn(num_gaussians, device=device) * 0.5
            return torch.stack([mu_h, mu_w], dim=1)   # [N, 2]，(h, w)

    # ------------------------------------------------------------------
    def rebuild_indexer(self):
        with torch.no_grad():
            log_s = self.log_scale.clamp(min=np.log(1), max=np.log(self.H / 6))
            self._indexer_cache = GaussianSpatialIndexer(
                self.mu.detach().clone(),
                torch.exp(log_s).detach().clone(),
                self.rotation.detach().clone(),
                (self.H, self.W),
                self.tile_size
            )

    def compute_covariance_inv_and_R(self):
        log_s = self.log_scale.clamp(min=np.log(1), max=np.log(self.H / 6))
        s     = torch.exp(log_s)
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

    # ------------------------------------------------------------------
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

        # Jacobian（无散参数化）
        #   J_local = [[ a,  b ],   ∂vel_h/∂(h,w)
        #              [ c, -a ]]   ∂vel_w/∂(h,w)   → 散度 = a+(-a) = 0
        if _has_jacobian(self):
            a, b, c, d = self.J_params.unbind(-1)
            J_local = torch.stack([
                torch.stack([ a, b], dim=-1),
                torch.stack([ c, d], dim=-1)
            ], dim=-2)
            J = torch.bmm(R, torch.bmm(J_local, R.transpose(1, 2)))
        else:
            J = None

        tile_counts     = indexer.cell_ptr[1:] - indexer.cell_ptr[:-1]
        active_tile_ids = torch.where(tile_counts > 0)[0]
        T = active_tile_ids.shape[0]
        _zf = torch.zeros(2, self.H, self.W, device=device)
        _zw = torch.zeros(1, self.H, self.W, device=device)
        if T == 0:
            zeros_hw = torch.zeros(self.H, self.W, device=device)
            return (_zf, _zw, zeros_hw, zeros_hw) if compute_derivatives else (_zf, _zw, None, None)

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

        support_mask = (
            (term <= 16.0)
            & valid_gmask.unsqueeze(1)
            & pixel_valid.unsqueeze(2)
        )
        
        gauss_w = torch.zeros_like(term)
        gauss_w[support_mask] = torch.exp(-0.5 * term[support_mask])
        
        weights = gauss_w * cur_opacity.squeeze(-1).unsqueeze(1)
        
        # gauss_w = (torch.exp(-0.5 * term)
        #            * torch.sigmoid(-2.0 * (term - 14.0)))
        # gauss_w = gauss_w * valid_gmask.unsqueeze(1) * pixel_valid.unsqueeze(2)
        # weights = gauss_w * cur_opacity.squeeze(-1).unsqueeze(1)   # [T, P, K]

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
            neg_Si_d  = torch.stack([neg_Si_dh, neg_Si_dw], dim=-1)   # [T, P, K, 2]

            J_contrib = cur_J if cur_J is not None else \
                        torch.zeros(T, K_max, 2, 2, device=device)

            outer  = vel_all.unsqueeze(-1) * neg_Si_d.unsqueeze(-2)
            dN_d   = w3.unsqueeze(-1) * (J_contrib.unsqueeze(1) + outer)   # [T, P, K, 2, 2]
            dW_d   = w3.squeeze(-1).unsqueeze(-1) * neg_Si_d

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

        # ── 普通路径 ───────────────────────────────────────────────────────
        if device.type == 'cuda' and HAS_GAUSSIAN_RENDER_CUDA:
            J_render = J if J is not None else torch.zeros(
                self.num_gaussians, 2, 2, device=device, dtype=self.mu.dtype
            )
            num_h_map, num_w_map, den_map = gaussian_render_cuda(
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

            # if not hasattr(self, "_printed_cuda_render"):
            #     print("[CUDA render] using custom gaussian_render_cuda")
            #     self._printed_cuda_render = True
            return flow, weight_map, None, None

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
    def render_at_resolution(self, eval_H: int, eval_W: int,
                             compute_derivatives: bool = True):
        """
        在任意目标分辨率 (eval_H, eval_W) 上直接光栅化高斯场。

        **所有高斯参数保持不变**——高斯定义在训练坐标空间 [0,H)×[0,W) 中，
        eval 时只改变查询网格的密度：
            - 训练时每格查询整数坐标 0, 1, 2, ..., H-1
            - eval 时查询分数坐标 0, H/eval_H, 2H/eval_H, ..., H-H/eval_H
              即在同一坐标空间内插入更多采样点，天然得到更高分辨率的输出。

        Tile 索引器（Gaussian→tile 分配）基于 [0,H)×[0,W) 坐标，eval 时完全复用，
        无需重建。每个 tile 内的查询点数从 ts×ts 变为 pts_h×pts_w
        （pts_h = ceil(ts * eval_H/H)），分数偏移量为 [ih * H/eval_H, iw * W/eval_W]。
        散射目标从 H×W 的 flat buffer 换成 eval_H×eval_W 的 flat buffer：
            output_i = round(gauss_h * eval_H/H)，output_j = round(gauss_w * eval_W/W)

        仅应在 torch.no_grad() + model.eval() 上下文中调用。
        """
        device = self.mu.device
        sf_h   = eval_H / self.H   # 行方向缩放因子
        sf_w   = eval_W / self.W   # 列方向缩放因子

        # ── Gaussian 参数（完全不变）──────────────────────────────────────
        if self._indexer_cache is None:
            self.rebuild_indexer()
        indexer = self._indexer_cache

        Sigma_inv, R, _ = self.compute_covariance_inv_and_R()
        opacity    = torch.sigmoid(self.opacity_logit)
        vel_global = torch.matmul(R, self.local_velocity.unsqueeze(-1)).squeeze(-1)

        if _has_jacobian(self):
            a, b, c, d = self.J_params.unbind(-1)
            J_local = torch.stack([
                torch.stack([ a,  b], dim=-1),
                torch.stack([ c, d], dim=-1)
            ], dim=-2)
            J = torch.bmm(R, torch.bmm(J_local, R.transpose(1, 2)))
        else:
            J = None

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
        support_mask = (
            (term <= 16.0)
            & valid_gmask.unsqueeze(1)
            & pixel_valid.unsqueeze(2)
        )
        
        gauss_w = torch.zeros_like(term)
        gauss_w[support_mask] = torch.exp(-0.5 * term[support_mask])
        
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
            neg_Si_d  = torch.stack([neg_Si_dh, neg_Si_dw], dim=-1)

            J_contrib = cur_J if cur_J is not None else \
                        torch.zeros(T, K_max, 2, 2, device=device)
            outer  = vel_all.unsqueeze(-1) * neg_Si_d.unsqueeze(-2)
            dN_d   = w3.unsqueeze(-1) * (J_contrib.unsqueeze(1) + outer)
            dW_d   = w3.squeeze(-1).unsqueeze(-1) * neg_Si_d

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


