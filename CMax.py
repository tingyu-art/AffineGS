try:
    from accelerators.cmax_fused import fused_warp_iwe_cuda
    _HAS_FUSED_WARP_IWE = True
    _FUSED_WARP_IWE_IMPORT_ERROR = None
    print("CMax fused extension: OK")
except Exception as exc:
    fused_warp_iwe_cuda = None
    _HAS_FUSED_WARP_IWE = False
    _FUSED_WARP_IWE_IMPORT_ERROR = repr(exc)
    print(f"CMax fused extension: NO ({_FUSED_WARP_IWE_IMPORT_ERROR})")

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class ContrastMaximizationLoss(nn.Module):
    """
    Contrast Maximization Loss for Event-based Optical Flow.

    坐标约定（贯穿全类）：
        事件格式  : [h, w, t, p]，h=行坐标，w=列坐标
        flow 格式 : [2, H, W]
                    通道 0 = vel_h（竖直，h 轴正方向）
                    通道 1 = vel_w（水平，w 轴正方向）

    平滑方式：
        - sigma       : 控制高斯宽度（贡献的"软硬"程度），推荐范围 [0.5, 1.5]
        - kernel_size : 控制 K×K 邻域大小（splat_radius = kernel_size // 2）
                        如 kernel_size=3 → 3×3 邻域，kernel_size=5 → 5×5 邻域
    """

    def __init__(self, loss_type='structure', sigma=0.5, kernel_size=3):
        super(ContrastMaximizationLoss, self).__init__()
        self.loss_type = loss_type
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.splat_radius = max(1, kernel_size // 2)

        _radius = self.splat_radius
        offs = torch.arange(-_radius, _radius + 1, dtype=torch.float32)
        dh_2d, dw_2d = torch.meshgrid(offs, offs, indexing='ij')
        self.register_buffer('_splat_dh',      dh_2d.reshape(-1))
        self.register_buffer('_splat_dw',      dw_2d.reshape(-1))
        self.register_buffer('_splat_dh_long', dh_2d.reshape(-1).long())
        self.register_buffer('_splat_dw_long', dw_2d.reshape(-1).long())

        _sigma = sigma if sigma > 0 else 0.6
        self._gauss_inv_scale = -1.0 / (2.0 * _sigma ** 2)

        sobel_w = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_h = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_w', sobel_w)
        self.register_buffer('sobel_h', sobel_h)

        self._vis_cache = {
            'input_pos': None, 'input_neg': None,
            'warped_pos': None, 'warped_neg': None,
            'flow': None, 'loss': 0.0
        }

    # ------------------------------------------------------------------
    # 基础工具
    # ------------------------------------------------------------------
    def _resolve_t_ref(self, events, t_ref=None):
        if t_ref is not None:
            if torch.is_tensor(t_ref):
                return t_ref.to(device=events.device, dtype=events.dtype)
            return torch.tensor(float(t_ref), device=events.device, dtype=events.dtype)

        t_min, t_max = events[:, 2].min(), events[:, 2].max()
        return 0.5 * (t_min + t_max)

    def _apply_loss(self, iwe):
        if self.loss_type == 'variance':
            return self.forward_variance_loss(iwe)
        elif self.loss_type == 'l2_max':
            return self.forward_l2_maximization(iwe)
        elif self.loss_type == 'gradient_inverse':
            return self.forward_gradient_inverse_loss(iwe)
        elif self.loss_type == 'structure':
            return self.forward_structure_loss(iwe)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _compute_iwe_from_events(self, events, image_size):
        if events is None or len(events) == 0:
            H, W = image_size
            return torch.zeros((H, W), device=self._splat_dh.device)
        return self.create_iwe(events, image_size)

    def _compute_iwe_from_events_separate_polarity(self, events, image_size):
        H, W = image_size
        if events is None or len(events) == 0:
            z = torch.zeros((H, W), device=self._splat_dh.device)
            return z, z

        mask_pos = events[:, 3] > 0.5
        mask_neg = ~mask_pos
        iwe_pos = self.create_iwe(events[mask_pos], image_size) if mask_pos.any() else torch.zeros((H, W), device=events.device)
        iwe_neg = self.create_iwe(events[mask_neg], image_size) if mask_neg.any() else torch.zeros((H, W), device=events.device)
        return iwe_pos, iwe_neg

    def _count_events_per_block(self, events, image_size, block_size):
        H, W = image_size
        Hb = H // block_size
        Wb = W // block_size
        if Hb == 0 or Wb == 0:
            raise ValueError(f"block_size={block_size} 过大，图像尺寸为 {H}x{W}")

        counts = torch.zeros((Hb, Wb), device=events.device, dtype=torch.float32)
        if events is None or len(events) == 0:
            return counts

        h = events[:, 0].long().clamp(0, H - 1)
        w = events[:, 1].long().clamp(0, W - 1)
        bh = (h // block_size).clamp(0, Hb - 1)
        bw = (w // block_size).clamp(0, Wb - 1)
        flat = bh * Wb + bw
        ones = torch.ones_like(flat, dtype=torch.float32)
        counts.view(-1).scatter_add_(0, flat, ones)
        return counts

    def _blockify(self, x, block_size):
        H, W = x.shape
        Hb = H // block_size
        Wb = W // block_size
        if Hb == 0 or Wb == 0:
            raise ValueError(f"block_size={block_size} 过大，图像尺寸为 {H}x{W}")
        x = x[:Hb * block_size, :Wb * block_size]
        blocks = x.view(Hb, block_size, Wb, block_size).permute(0, 2, 1, 3).reshape(Hb * Wb, -1)
        return blocks, Hb, Wb

    def _compute_block_difference(self, iwe_a, iwe_b, block_size=8, eps=1e-6):
        blocks_a, Hb, Wb = self._blockify(iwe_a, block_size)
        blocks_b, _, _ = self._blockify(iwe_b, block_size)

        sum_a = blocks_a.sum(dim=1, keepdim=True)
        sum_b = blocks_b.sum(dim=1, keepdim=True)
        norm_a = blocks_a / (sum_a + eps)
        norm_b = blocks_b / (sum_b + eps)
        score = (norm_a - norm_b).abs().mean(dim=1)
        return score.view(Hb, Wb)

    def _compute_local_variance_score(self, iwe, window_size=11, eps=1e-6):
        """
        计算局部方差 / 局部区域 IWE 和。
        输出越小表示局部聚焦越差，更可能对应较大误差。
        """
        if window_size <= 1:
            window_size = 3
        if window_size % 2 == 0:
            window_size = window_size + 1

        x = iwe.view(1, 1, iwe.shape[0], iwe.shape[1])
        pad = window_size // 2
        kernel_mean = torch.ones(
            (1, 1, window_size, window_size),
            device=iwe.device,
            dtype=iwe.dtype,
        ) / float(window_size * window_size)
        kernel_sum = torch.ones(
            (1, 1, window_size, window_size),
            device=iwe.device,
            dtype=iwe.dtype,
        )

        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        mu = F.conv2d(x_pad, kernel_mean)
        resid2 = (x - mu) ** 2
        resid2_pad = F.pad(resid2, (pad, pad, pad, pad), mode='reflect')
        var_map = F.conv2d(resid2_pad, kernel_mean)
        local_sum = F.conv2d(x_pad, kernel_sum).clamp_min(0.0)
        score_map = var_map / (local_sum + eps)
        return score_map.squeeze(0).squeeze(0), var_map.squeeze(0).squeeze(0), local_sum.squeeze(0).squeeze(0)

    def _compute_block_local_variance_score(self, iwe, block_size=8, eps=1e-6):
        window_size = block_size if block_size % 2 == 1 else (block_size + 1)
        score_map, var_map, local_sum = self._compute_local_variance_score(
            iwe,
            window_size=window_size,
            eps=eps,
        )
        blocks, Hb, Wb = self._blockify(score_map, block_size)
        block_score = blocks.mean(dim=1).view(Hb, Wb)
        return block_score, score_map, var_map, local_sum

    # ------------------------------------------------------------------
    # 原有 warp / IWE
    # ------------------------------------------------------------------
    def sample_flow_at_locations(self, flow, h, w, image_size):
        H, W = image_size
        w_norm = 2.0 * w / (W - 1) - 1.0
        h_norm = 2.0 * h / (H - 1) - 1.0
        grid = torch.stack([w_norm, h_norm], dim=-1).view(1, 1, -1, 2)

        if flow.dim() == 3:
            flow = flow.unsqueeze(0)

        sampled = F.grid_sample(
            input=flow,
            grid=grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )
        return sampled.squeeze(0).squeeze(1).t()

    def warp_events(self, events, flow, t_ref, image_size):
        h, w, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]

        flow_at_events = self.sample_flow_at_locations(flow, h, w, image_size)
        vel_h_at = flow_at_events[:, 0]
        vel_w_at = flow_at_events[:, 1]

        dt = t_ref - t
        h_warped = h + vel_h_at * dt
        w_warped = w + vel_w_at * dt

        return torch.stack([h_warped, w_warped, t, p], dim=1)

    def create_iwe(self, warped_events, image_size):
        """
        完全可微的 IWE 构建（Differentiable Soft Splatting）。

        平滑参数由构造函数控制：
            self.sigma        → 高斯宽度（贡献软硬程度）
            self.splat_radius → 邻域半径（= kernel_size // 2）

        梯度流路径：
            flow → grid_sample → vel → h_warped/w_warped → gauss_w = exp(-dist²) → scatter_add → iwe
        """
        H, W = image_size
        device = warped_events.device

        h_warped = warped_events[:, 0]
        w_warped = warped_events[:, 1]

        margin = 1.0
        boundary_weight = (
            torch.sigmoid((h_warped - (-margin)) * 5.0) *
            torch.sigmoid(-(h_warped - (H - 1 + margin)) * 5.0) *
            torch.sigmoid((w_warped - (-margin)) * 5.0) *
            torch.sigmoid(-(w_warped - (W - 1 + margin)) * 5.0)
        )

        with torch.no_grad():
            hard_mask = boundary_weight.detach() > 1e-4
        if hard_mask.sum() == 0:
            return torch.zeros((H, W), device=device)

        h_w = h_warped[hard_mask]
        w_w = w_warped[hard_mask]
        bw = boundary_weight[hard_mask]

        dh = self._splat_dh
        dw = self._splat_dw

        h_floor = torch.floor(h_w)
        w_floor = torch.floor(w_w)

        frac_h = h_w - h_floor
        frac_w = w_w - w_floor
        dist2 = (frac_h.unsqueeze(1) - dh.unsqueeze(0)) ** 2 + \
                (frac_w.unsqueeze(1) - dw.unsqueeze(0)) ** 2

        gauss_w = torch.exp(dist2 * self._gauss_inv_scale)
        gauss_w = gauss_w * bw.unsqueeze(1)

        with torch.no_grad():
            ph = h_floor.long().unsqueeze(1) + self._splat_dh_long.unsqueeze(0)
            pw = w_floor.long().unsqueeze(1) + self._splat_dw_long.unsqueeze(0)
            ph_c = ph.clamp(0, H - 1)
            pw_c = pw.clamp(0, W - 1)
            flat_idx = ph_c * W + pw_c

        iwe_flat = torch.zeros(H * W, device=device)
        iwe_flat.scatter_add_(0, flat_idx.reshape(-1), gauss_w.reshape(-1))
        iwe = iwe_flat.reshape(H, W)

        return iwe

    def compute_gradient_magnitude(self, iwe):
        iwe_4d = iwe.view(1, 1, iwe.shape[0], iwe.shape[1])
        grad_w = F.conv2d(iwe_4d, self.sobel_w, padding=1)
        grad_h = F.conv2d(iwe_4d, self.sobel_h, padding=1)
        return torch.abs(grad_w).squeeze() + torch.abs(grad_h).squeeze()

    # ------------------------------------------------------------------
    # 新增：块级时序对齐残差 proxy（不加速）
    # ------------------------------------------------------------------
    def compute_block_temporal_residual(
        self,
        events,
        flow,
        image_size,
        block_size=16,
        t_ref=None,
        min_events_per_block=0,
        use_polarity=False,
        no_grad=True,
    ):
        """
        计算块级局部方差 proxy（无监督 proxy，不参与主数据损失）。

        原理：
            - 使用当前 flow 将整个事件窗口 warp 到同一 t_ref；
            - 从 warped IWE 计算局部方差 / 局部区域 IWE 和；
            - 再在块级求均值，得到每个块的局部聚焦质量；
            - 数值越小，说明该块局部聚焦越差，更可能对应较大误差。

        参数：
            events : [N,4]，格式 [h, w, t, p]
            flow   : [2,H,W]
            image_size : (H, W)
            block_size : 像素块大小，例如 8 表示 8x8 block
            t_ref  : 参考时刻；None 时默认取当前事件窗口中心时刻
            min_events_per_block : 用于构造 valid_map 的最小事件数阈值
            use_polarity : True 时分别对正/负事件构建 IWE，并对两个 polarity 的
                           局部方差 score 取平均；False 时忽略极性，统一构建 IWE
            return_debug : True 时返回中间结果字典
            no_grad : True 时在 torch.no_grad() 下执行，适合密度控制阶段调用

        返回：
            score_map : [Hb, Wb]，越小表示该块局部聚焦越差
            valid_map : [Hb, Wb] bool
            debug     : 可选调试字典
        """
        def _impl():
            events_local = events
            if events_local is None or len(events_local) == 0:
                H, W = image_size
                Hb = H // block_size
                Wb = W // block_size
                score = torch.zeros((Hb, Wb), device=flow.device, dtype=flow.dtype)
                valid = torch.zeros((Hb, Wb), device=flow.device, dtype=torch.bool)
                return (score, valid, {}) if return_debug else (score, valid)

            events_local = events_local.to(device=flow.device, dtype=flow.dtype)
            t_ref_local = self._resolve_t_ref(events_local, t_ref=t_ref)

            H, W = image_size
            Hb = H // block_size
            Wb = W // block_size
            if Hb == 0 or Wb == 0:
                raise ValueError(f"block_size={block_size} 过大，图像尺寸为 {H}x{W}")

            warped_events = self.warp_events(events_local, flow, t_ref_local, image_size)

            if use_polarity:
                iwe_pos, iwe_neg = self._compute_iwe_from_events_separate_polarity(warped_events, image_size)
                score_pos_block, score_pos_map, var_pos_map, sum_pos_map = self._compute_block_local_variance_score(
                    iwe_pos,
                    block_size=block_size,
                )
                score_neg_block, score_neg_map, var_neg_map, sum_neg_map = self._compute_block_local_variance_score(
                    iwe_neg,
                    block_size=block_size,
                )
                score_map = 0.5 * (score_pos_block + score_neg_block)
            else:
                iwe_all = self.create_iwe(warped_events, image_size)
                score_map, score_full_map, var_map, local_sum_map = self._compute_block_local_variance_score(
                    iwe_all,
                    block_size=block_size,
                )

            count_total = self._count_events_per_block(events_local, image_size, block_size)
            valid_map = count_total >= float(min_events_per_block)

            return score_map, valid_map

        if no_grad:
            with torch.no_grad():
                return _impl()
        return _impl()

    # --- Loss Functions ---
    def forward_variance_loss(self, iwe):
        active_mask = iwe > 1e-5
        if active_mask.sum() > 10:
            return -torch.var(iwe[active_mask])
        return -torch.var(iwe)

    def forward_l2_maximization(self, iwe):
        iwe_mean = torch.mean(iwe)
        iwe_sq_mean = torch.mean(iwe ** 2)
        return -iwe_sq_mean / (iwe_mean.detach() ** 2 + 1e-8)

    def forward_structure_loss(self, iwe):
        iwe4 = iwe.unsqueeze(0).unsqueeze(0)
        gw = F.conv2d(iwe4, self.sobel_w, padding=1)
        gh = F.conv2d(iwe4, self.sobel_h, padding=1)
        grad_energy = torch.mean(gw ** 2 + gh ** 2)
        intensity_energy = torch.mean(iwe ** 2)
        return -grad_energy / (intensity_energy + 1e-6)

    def forward_gradient_inverse_loss(self, iwe):
        grad_mag = self.compute_gradient_magnitude(iwe)
        return 1.0 / (torch.mean(grad_mag) + 1e-6)

    def forward(self, events, flow, image_size, vis=False):
        """
        events : [N, 4]，[:,0]=h，[:,1]=w，[:,2]=t，[:,3]=p
        flow   : [2, H, W]，通道0=vel_h，通道1=vel_w
        """
        t_ref = self._resolve_t_ref(events)

        use_fused = (
            _HAS_FUSED_WARP_IWE and
            events.is_cuda and
            flow.is_cuda and
            events.dtype == torch.float32 and
            flow.dtype == torch.float32
        )

        if use_fused:
            iwe_2ch = fused_warp_iwe_cuda(
                events.contiguous(),
                flow.contiguous(),
                image_size,
                float(t_ref.item()),
                float(self.sigma),
                int(self.kernel_size),
            )
            iwe_neg = iwe_2ch[0]
            iwe_pos = iwe_2ch[1]

            loss_total = 0.0

            if vis:
                self._vis_cache['flow'] = flow.detach().cpu()
                self._vis_cache['warped_pos'] = iwe_pos.detach().cpu()
                self._vis_cache['warped_neg'] = iwe_neg.detach().cpu()

                mask_pos = (events[:, 3] > 0.5)
                mask_neg = (events[:, 3] <= 0.5)

                if len(events[mask_pos]) > 0:
                    self._vis_cache['input_pos'] = self.create_iwe(events[mask_pos], image_size).detach().cpu()
                if len(events[mask_neg]) > 0:
                    self._vis_cache['input_neg'] = self.create_iwe(events[mask_neg], image_size).detach().cpu()

            if iwe_pos.abs().sum() > 0:
                loss_total = loss_total + self._apply_loss(iwe_pos)
            if iwe_neg.abs().sum() > 0:
                loss_total = loss_total + self._apply_loss(iwe_neg)

            if vis:
                self._vis_cache['loss'] = (
                    loss_total.item() if isinstance(loss_total, torch.Tensor) else loss_total
                )
            return loss_total

        warped_events = self.warp_events(events, flow, t_ref, image_size)

        mask_pos = (events[:, 3] > 0.5)
        mask_neg = (events[:, 3] <= 0.5)

        loss_total = 0.0

        def calculate_loss(w_events, tag):
            if len(w_events) > 0:
                iwe = self.create_iwe(w_events, image_size)
                if vis:
                    self._vis_cache[tag] = iwe.detach().cpu()
                return self._apply_loss(iwe)
            return 0.0

        loss_total += calculate_loss(warped_events[mask_pos], 'warped_pos')
        loss_total += calculate_loss(warped_events[mask_neg], 'warped_neg')

        if vis:
            self._vis_cache['flow'] = flow.detach().cpu()
            if len(events[mask_pos]) > 0:
                self._vis_cache['input_pos'] = self.create_iwe(events[mask_pos], image_size).detach().cpu()
            if len(events[mask_neg]) > 0:
                self._vis_cache['input_neg'] = self.create_iwe(events[mask_neg], image_size).detach().cpu()
            self._vis_cache['loss'] = (
                loss_total.item() if isinstance(loss_total, torch.Tensor) else loss_total
            )

        return loss_total

    def plot_visualization(self,
                           grid_step=20,
                           arrow_scale=None,
                           grid_loss_n=None,
                           save_dir=None,
                           save_prefix="vis",
                           save_indices=(1, 2, 3, 4, 5)):
        cache = self._vis_cache
        if cache['input_pos'] is None:
            print("No visualization data.")
            return

        def to_numpy(img):
            if img is None:
                return None
            if isinstance(img, torch.Tensor):
                return img.detach().cpu().numpy()
            return np.asarray(img)

        def save_clean_image(img, save_path, cmap='hot'):
            arr = to_numpy(img)
            if arr is None:
                return

            vmin = np.nanmin(arr)
            vmax = np.nanmax(arr)
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-8

            plt.imsave(save_path, arr, cmap=cmap, origin='upper', vmin=vmin, vmax=vmax)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            save_map = {
                1: (cache['input_pos'],  "input_pos",  "hot"),
                2: (cache['warped_pos'], "warped_pos", "hot"),
                4: (cache['input_neg'],  "input_neg",  "viridis"),
                5: (cache['warped_neg'], "warped_neg", "viridis"),
            }

            for idx in save_indices:
                if idx in save_map:
                    img, name, cmap = save_map[idx]
                    if img is not None:
                        save_path = os.path.join(save_dir, f"{save_prefix}_{idx}_{name}.svg")
                        save_clean_image(img, save_path, cmap=cmap)

        n_rows = 3 if grid_loss_n is not None else 2
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        fig.suptitle(f"Total Loss: {cache['loss']:.4f}", fontsize=14)

        def plot_im(ax, img, title, cmap='hot'):
            if img is not None:
                arr = to_numpy(img)
                im = ax.imshow(arr, cmap=cmap, origin='upper')
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(title)
            ax.axis('off')

        plot_im(axes[0, 0], cache['input_pos'],  "Input Events (+)")
        plot_im(axes[1, 0], cache['input_neg'],  "Input Events (-)")
        plot_im(axes[0, 1], cache['warped_pos'], "Warped IWE (+)")
        plot_im(axes[1, 1], cache['warped_neg'], "Warped IWE (-)")

        flow = cache['flow']
        if flow is not None:
            flow_np = to_numpy(flow)
            vel_h = flow_np[0]
            vel_w = flow_np[1]
            H, W = vel_h.shape

            ax_quiver = axes[0, 2]
            h_grid, w_grid = np.mgrid[0:H:grid_step, 0:W:grid_step]
            vel_h_s = vel_h[::grid_step, ::grid_step]
            vel_w_s = vel_w[::grid_step, ::grid_step]
            mag = np.sqrt(vel_h ** 2 + vel_w ** 2)

            ax_quiver.imshow(mag, cmap='gray', alpha=0.3, origin='upper')
            ax_quiver.quiver(
                w_grid, h_grid, vel_w_s, vel_h_s,
                color='red', angles='xy', scale_units='xy', scale=arrow_scale
            )
            ax_quiver.set_title(f"Flow Field (Step={grid_step})")
            ax_quiver.axis('off')

            ax_mag = axes[1, 2]
            ax_mag.imshow(mag, cmap='viridis', origin='upper')
            ax_mag.set_title("Flow Magnitude")
            ax_mag.axis('off')
        else:
            axes[0, 2].axis('off')
            axes[1, 2].axis('off')

        if grid_loss_n is not None:
            def compute_grid_l2max(iwe_tensor):
                if iwe_tensor is None:
                    return None
                iwe = to_numpy(iwe_tensor)
                H, W = iwe.shape
                gh = H // grid_loss_n
                gw = W // grid_loss_n
                if gh == 0 or gw == 0:
                    print(f"grid_loss_n={grid_loss_n} 过大，图像尺寸 {H}x{W} 不够切块")
                    return None

                grid_map = np.zeros((gh, gw), dtype=np.float32)
                for i in range(gh):
                    for j in range(gw):
                        patch = iwe[
                            i * grid_loss_n:(i + 1) * grid_loss_n,
                            j * grid_loss_n:(j + 1) * grid_loss_n
                        ]
                        mean_sq = np.mean(patch ** 2)
                        mean_val = np.mean(patch)
                        grid_map[i, j] = mean_sq / (mean_val ** 2 + 1e-8)
                return grid_map

            def compute_alignment_ratio(input_iwe, warped_iwe):
                g_input = compute_grid_l2max(input_iwe)
                g_warped = compute_grid_l2max(warped_iwe)
                if g_input is None or g_warped is None:
                    return None
                return g_warped / (g_input + 1e-8)

            ratio_pos = compute_alignment_ratio(cache['input_pos'], cache['warped_pos'])
            ratio_neg = compute_alignment_ratio(cache['input_neg'], cache['warped_neg'])

            def plot_ratio(ax, ratio, title):
                if ratio is None:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    ax.set_title(title)
                    ax.axis('off')
                    return

                vmax = max(abs(ratio - 1.0).max() * 1.2, 0.5)
                im = ax.imshow(
                    ratio, cmap='RdYlGn', origin='upper',
                    vmin=1.0 - vmax, vmax=1.0 + vmax
                )
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.axis('off')

            plot_ratio(axes[2, 0], ratio_pos,
                       f"Alignment Ratio (+) n={grid_loss_n}\ngreen=good red=bad")
            plot_ratio(axes[2, 1], ratio_neg,
                       f"Alignment Ratio (-) n={grid_loss_n}\ngreen=good red=bad")
            axes[2, 2].axis('off')

        plt.tight_layout()
        plt.show()
