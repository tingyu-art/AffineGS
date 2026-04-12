try:
    from cmax_fused import fused_warp_iwe_cuda
    _HAS_FUSED_WARP_IWE = True
    print("CUDA acceleration: OK")
except Exception:
    fused_warp_iwe_cuda = None
    _HAS_FUSED_WARP_IWE = False
    print("CUDA acceleration: NO")

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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

        # float 版用于 dist2 计算，long 版用于像素索引计算。
        _radius = self.splat_radius
        offs = torch.arange(-_radius, _radius + 1, dtype=torch.float32)
        dh_2d, dw_2d = torch.meshgrid(offs, offs, indexing='ij')
        self.register_buffer('_splat_dh',      dh_2d.reshape(-1))        # [K²] float
        self.register_buffer('_splat_dw',      dw_2d.reshape(-1))        # [K²] float
        self.register_buffer('_splat_dh_long', dh_2d.reshape(-1).long()) # [K²] long
        self.register_buffer('_splat_dw_long', dw_2d.reshape(-1).long()) # [K²] long

        # ── 优化②：预算高斯缩放系数 ─────────────────────────────────────────
        _sigma = sigma if sigma > 0 else 0.6
        self._gauss_inv_scale = -1.0 / (2.0 * _sigma ** 2)   # 标量，forward 直接乘

        # 预注册 Sobel 算子
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

    def sample_flow_at_locations(self, flow, h, w, image_size):
        H, W = image_size
        w_norm = 2.0 * w / (W - 1) - 1.0
        h_norm = 2.0 * h / (H - 1) - 1.0
        grid = torch.stack([w_norm, h_norm], dim=-1).view(1, 1, -1, 2)

        if flow.dim() == 3:
            flow = flow.unsqueeze(0)

        sampled = F.grid_sample(input=flow, grid=grid, mode='bilinear', align_corners=True, padding_mode='border')
        
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

        # 软边界过滤
        margin = 1.0
        boundary_weight = (
            torch.sigmoid( (h_warped - (-margin))        * 5.0) *
            torch.sigmoid(-(h_warped - (H - 1 + margin)) * 5.0) *
            torch.sigmoid( (w_warped - (-margin))        * 5.0) *
            torch.sigmoid(-(w_warped - (W - 1 + margin)) * 5.0)
        )

        with torch.no_grad():
            hard_mask = boundary_weight.detach() > 1e-4
        if hard_mask.sum() == 0:
            return torch.zeros((H, W), device=device)

        h_w = h_warped[hard_mask]
        w_w = w_warped[hard_mask]
        bw  = boundary_weight[hard_mask]

        # ── Soft Splatting ──────────────────────────────────────────────────
        dh = self._splat_dh   # [K²] float，已在正确 device
        dw = self._splat_dw   # [K²] float

        # floor 计算一次，复用两处
        h_floor = torch.floor(h_w)   # [M]
        w_floor = torch.floor(w_w)   # [M]

        # frac = x - floor(x)
        frac_h = h_w - h_floor   # [M]
        frac_w = w_w - w_floor   # [M]
        dist2 = (frac_h.unsqueeze(1) - dh.unsqueeze(0)) ** 2 + \
                (frac_w.unsqueeze(1) - dw.unsqueeze(0)) ** 2   # [M, K²]

        # _gauss_inv_scale = -1/(2σ²)
        gauss_w = torch.exp(dist2 * self._gauss_inv_scale)   # [M, K²]
        gauss_w = gauss_w * bw.unsqueeze(1)

        with torch.no_grad():
            ph = h_floor.long().unsqueeze(1) + self._splat_dh_long.unsqueeze(0)   # [M, K²]
            pw = w_floor.long().unsqueeze(1) + self._splat_dw_long.unsqueeze(0)   # [M, K²]
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

    # --- Loss Functions ---
    def forward_variance_loss(self, iwe):
        active_mask = iwe > 1e-5
        if active_mask.sum() > 10:
            return -torch.var(iwe[active_mask])
        return -torch.var(iwe)

    def forward_l2_maximization(self, iwe):   # 现在使用的
        iwe_mean    = torch.mean(iwe)
        iwe_sq_mean = torch.mean(iwe ** 2)
        return -iwe_sq_mean / (iwe_mean.detach() ** 2 + 1e-8)

    def forward_structure_loss(self, iwe):
        iwe4 = iwe.unsqueeze(0).unsqueeze(0)
        gw = F.conv2d(iwe4, self.sobel_w, padding=1)
        gh = F.conv2d(iwe4, self.sobel_h, padding=1)
        grad_energy      = torch.mean(gw ** 2 + gh ** 2)
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

        t_min, t_max = events[:, 2].min(), events[:, 2].max()
        t_ref = 0.5 * (t_min + t_max)
    
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
                    self._vis_cache['input_pos'] = \
                        self.create_iwe(events[mask_pos], image_size).detach().cpu()
                if len(events[mask_neg]) > 0:
                    self._vis_cache['input_neg'] = \
                        self.create_iwe(events[mask_neg], image_size).detach().cpu()
    
            def apply_loss(iwe):
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
    
            if iwe_pos.abs().sum() > 0:
                loss_total = loss_total + apply_loss(iwe_pos)
            if iwe_neg.abs().sum() > 0:
                loss_total = loss_total + apply_loss(iwe_neg)
    
            if vis:
                self._vis_cache['loss'] = (
                    loss_total.item() if isinstance(loss_total, torch.Tensor) else loss_total
                )
            return loss_total
    
        # ---------------- fallback: 原来的 PyTorch 版本 ----------------
        warped_events = self.warp_events(events, flow, t_ref, image_size)
    
        mask_pos = (events[:, 3] > 0.5)
        mask_neg = (events[:, 3] <= 0.5)
    
        loss_total = 0.0
    
        def calculate_loss(w_events, tag):
            if len(w_events) > 0:
                iwe = self.create_iwe(w_events, image_size)
                if vis:
                    self._vis_cache[tag] = iwe.detach().cpu()
                if self.loss_type == 'variance':
                    return self.forward_variance_loss(iwe)
                elif self.loss_type == 'l2_max':
                    return self.forward_l2_maximization(iwe)
                elif self.loss_type == 'gradient_inverse':
                    return self.forward_gradient_inverse_loss(iwe)
                elif self.loss_type == 'structure':
                    return self.forward_structure_loss(iwe)
            return 0.0
    
        loss_total += calculate_loss(warped_events[mask_pos], 'warped_pos')
        loss_total += calculate_loss(warped_events[mask_neg], 'warped_neg')
    
        if vis:
            self._vis_cache['flow'] = flow.detach().cpu()
            if len(events[mask_pos]) > 0:
                self._vis_cache['input_pos'] = \
                    self.create_iwe(events[mask_pos], image_size).detach().cpu()
            if len(events[mask_neg]) > 0:
                self._vis_cache['input_neg'] = \
                    self.create_iwe(events[mask_neg], image_size).detach().cpu()
            self._vis_cache['loss'] = (
                loss_total.item() if isinstance(loss_total, torch.Tensor) else loss_total
            )
    
        return loss_total

    def plot_visualization(self, grid_step=20, arrow_scale=None, grid_loss_n=None):
        cache = self._vis_cache
        if cache['input_pos'] is None:
            print("No visualization data.")
            return

        n_rows = 3 if grid_loss_n is not None else 2
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        fig.suptitle(f"Total Loss: {cache['loss']:.4f}", fontsize=14)

        def plot_im(ax, img, title, cmap='hot'):
            if img is not None:
                im = ax.imshow(img.numpy() if isinstance(img, torch.Tensor) else img,
                               cmap=cmap, origin='upper')
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
            vel_h = flow[0].numpy()
            vel_w = flow[1].numpy()
            H, W  = vel_h.shape

            ax_quiver = axes[0, 2]
            h_grid, w_grid = np.mgrid[0:H:grid_step, 0:W:grid_step]
            vel_h_s = vel_h[::grid_step, ::grid_step]
            vel_w_s = vel_w[::grid_step, ::grid_step]
            mag = np.sqrt(vel_h ** 2 + vel_w ** 2)
            ax_quiver.imshow(mag, cmap='gray', alpha=0.3, origin='upper')
            ax_quiver.quiver(w_grid, h_grid, vel_w_s, vel_h_s,
                             color='red', angles='xy', scale_units='xy',
                             scale=arrow_scale)
            ax_quiver.set_title(f"Flow Field (Step={grid_step})")
            ax_quiver.axis('off')

            ax_mag = axes[1, 2]
            im_mag = ax_mag.imshow(mag, cmap='viridis', origin='upper')
            ax_mag.set_title("Flow Magnitude")
            # plt.colorbar(im_mag, ax=ax_mag, fraction=0.046, pad=0.04)
            ax_mag.axis('off')
        else:
            axes[0, 2].axis('off')
            axes[1, 2].axis('off')

        if grid_loss_n is not None:

            def compute_grid_l2max(iwe_tensor):
                if iwe_tensor is None:
                    return None
                iwe  = iwe_tensor.numpy() if isinstance(iwe_tensor, torch.Tensor) else iwe_tensor
                H, W = iwe.shape
                gh   = H // grid_loss_n
                gw   = W // grid_loss_n
                if gh == 0 or gw == 0:
                    print(f"grid_loss_n={grid_loss_n} 过大，图像尺寸 {H}x{W} 不够切块")
                    return None
                grid_map = np.zeros((gh, gw))
                for i in range(gh):
                    for j in range(gw):
                        patch = iwe[i*grid_loss_n:(i+1)*grid_loss_n,
                                    j*grid_loss_n:(j+1)*grid_loss_n]
                        mean_sq  = np.mean(patch ** 2)
                        mean_val = np.mean(patch)
                        grid_map[i, j] = mean_sq / (mean_val ** 2 + 1e-8)
                return grid_map

            def compute_alignment_ratio(input_iwe, warped_iwe):
                g_input  = compute_grid_l2max(input_iwe)
                g_warped = compute_grid_l2max(warped_iwe)
                if g_input is None or g_warped is None:
                    return None
                return g_warped / (g_input + 1e-8)

            ratio_pos = compute_alignment_ratio(cache['input_pos'],  cache['warped_pos'])
            ratio_neg = compute_alignment_ratio(cache['input_neg'],  cache['warped_neg'])

            def plot_ratio(ax, ratio, title):
                if ratio is None:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    ax.set_title(title)
                    ax.axis('off')
                    return
                vmax = max(abs(ratio - 1.0).max() * 1.2, 0.5)
                im = ax.imshow(ratio, cmap='RdYlGn', origin='upper',
                               vmin=1.0 - vmax, vmax=1.0 + vmax)
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