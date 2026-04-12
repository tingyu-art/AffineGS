"""
坐标约定（贯穿全文件）：
    事件格式  : [h, w, t, p]，h=行坐标（竖直），w=列坐标（水平）
    flow 格式 : [2, H, W]
                通道 0 = vel_h（竖直，h 轴正方向）
                通道 1 = vel_w（水平，w 轴正方向）

公开接口（三层）
----------------
train_gaussian_flow(events, cfg, flowgt=None)
    → (model, history)
    只负责训练，返回训练完毕的模型和损失历史。

render_gaussian_flow(model, eval_H=None, eval_W=None, verbose=True)
    → GaussianFlowResult
    对已训练模型进行验证推理，高斯参数不变，直接以分数坐标查询任意分辨率网格。
    可对同一模型多次调用、输出不同分辨率，无需重复训练。

run_gaussian_flow(events, cfg, flowgt=None, eval_H=None, eval_W=None)
    → GaussianFlowResult
    便捷封装，内部依次调用上面两个函数，适合一次性使用场景。

用法示例
--------
from gaussian_flow import GaussianFlowConfig, train_gaussian_flow, render_gaussian_flow

cfg = GaussianFlowConfig()

# 训练一次
model, history = train_gaussian_flow(events, cfg)

# 任意分辨率渲染，无需重训练
result_256  = render_gaussian_flow(model, 256,  256)
result_512  = render_gaussian_flow(model, 512,  512)
result_1024 = render_gaussian_flow(model, 1024, 1024)

# result.flow      : numpy ndarray (2, eval_H, eval_W)
# result.weight_map: torch.Tensor  (1, eval_H, eval_W)
# result.div       : torch.Tensor  (eval_H, eval_W)，∂vel_h/∂h + ∂vel_w/∂w
# result.vorticity : torch.Tensor  (eval_H, eval_W)，∂vel_w/∂h − ∂vel_h/∂w
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional

from Cmax import ContrastMaximizationLoss
from utils.flow_classifier import is_turbulence
from gaussian_render_ext.gaussian_render import gaussian_render_cuda


# ─────────────────────────────────────────────────────────────────────────────
#  返回值 dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GaussianFlowResult:
    """
    run_gaussian_flow 的返回值。

    所有空间量均在验证推理阶段（训练结束后，torch.no_grad() + model.eval()）计算，
    分辨率为 eval_H × eval_W（可通过 run_gaussian_flow 的 eval_H/eval_W 参数指定）。

    flow       : np.ndarray (2, eval_H, eval_W)，通道0=vel_h，通道1=vel_w
    model      : 训练完成的 GaussianVelocityField_Tiled（在原设备上，保持 train() 状态）
    weight_map : torch.Tensor (1, eval_H, eval_W)，高斯权重图
    div        : torch.Tensor (eval_H, eval_W)，解析散度 ∂vel_h/∂h + ∂vel_w/∂w
    vorticity  : torch.Tensor (eval_H, eval_W)，解析旋度 ∂vel_w/∂h − ∂vel_h/∂w
    eval_H     : int，输出分辨率（行）
    eval_W     : int，输出分辨率（列）
    history    : dict，训练过程中各项损失和统计量
    """
    flow:       np.ndarray
    model:      object
    weight_map: torch.Tensor
    div:        torch.Tensor
    vorticity:  torch.Tensor
    eval_H:     int
    eval_W:     int
    history:    dict


# ─────────────────────────────────────────────────────────────────────────────
#  配置 dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GaussianFlowConfig:
    """所有超参数集中在此处，不改变原始代码中的默认值。"""

    # ── 分辨率 ──────────────────────────────────────────────────────────────
    H: int = 256                    # 流场分辨率（行）
    W: int = 256                    # 流场分辨率（列）

    # ── 模型初始化 ───────────────────────────────────────────────────────────
    init_n: int = 200               # 初始高斯数量
    tile_size: int = 32             # 分块大小
    density_alpha: float = 0.0      # 事件密度引导权重（0=均匀，1=纯密度）
    enable_jacobian: bool = True    # 是否启用雅可比矩阵参数

    # ── 训练控制 ─────────────────────────────────────────────────────────────
    iterations: int = 300           # 总迭代次数
    rebuild_interval: int = 10      # 重建空间索引的间隔
    density_interval: int = 50      # 密度控制（clone/split/prune）执行间隔
    phase: List[int] = field(default_factory=lambda: [50, 200, 250])   # 训练阶段分界点 [phase0_end, phase1_end, phase2_end]

    # ── 优化器 lr ────────────────────────────────────────────────────────────
    lr_mu: float = 2.0              # 均值参数学习率
    lr_j: Optional[float] = None   # Jacobian lr；None = 自动检测（湍流→0.03，其他→0.001）

    # ── 损失权重 ─────────────────────────────────────────────────────────────
    w_div: float = 5             # 散度正则化权重
    w_smooth: float = 0.01          # 涡量平滑正则化权重

    # ── log_scale clamp 范围 ─────────────────────────────────────────────────
    log_scale_min: float = np.log(0.5)

    # ── 密度控制参数 ─────────────────────────────────────────────────────────
    max_clone: int = 50
    max_split: int = 50
    prune_opacity_thresh: float = 0.1
    max_anisotropy: float = 3.0

    # ── 事件密度图平滑核 ─────────────────────────────────────────────────────
    density_smooth_k: int = 9
    clone_candidates_quantile: float = 0.2

    # ── Cmax 损失参数 ────────────────────────────────────────────────────────
    cmax_loss_type: str = 'l2_max'
    cmax_sigma: float = 0.6
    cmax_kernel_size: int = 5

    # ── 设备 ────────────────────────────────────────────────────────────────
    device: str = 'cuda'            # 'cuda' 或 'cpu'
    verbose: bool = False


# ─────────────────────────────────────────────────────────────────────────────
#  Sobel 核（模块级常量，按需移至 device）
# ─────────────────────────────────────────────────────────────────────────────

_SOBEL_W = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3).expand(2, -1, -1, -1)
_SOBEL_H = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3).expand(2, -1, -1, -1)

# ─────────────────────────────────────────────────────────────────────────────
#  GaussianSpatialIndexer（分块索引）
# ─────────────────────────────────────────────────────────────────────────────

class GaussianSpatialIndexer:
    """
    与原版接口完全兼容。
    build() 使用基于 bincount 的计数排序（O(M+T)），比 argsort O(M log M) 更快。
    """

    def __init__(self, means, scales, rotations, domain_size,
                 tile_size, truncation_sigma=4.0, max_tile_span=16):
        self.means            = means
        self.scales           = scales
        self.rotations        = rotations.reshape(-1, 1)
        self.H, self.W        = domain_size
        self.tile_size        = tile_size
        self.truncation_sigma = truncation_sigma
        self.max_tile_span    = max_tile_span
        self.device           = means.device

        self.grid_h    = (self.H + tile_size - 1) // tile_size
        self.grid_w    = (self.W + tile_size - 1) // tile_size
        self.num_tiles = self.grid_h * self.grid_w

        self.cell_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.cell_ptr     = torch.zeros(self.num_tiles + 1, dtype=torch.long,
                                        device=self.device)
        self.build()

    def build(self):
        N      = self.means.shape[0]
        device = self.device
        if N == 0:
            return

        cos_t = torch.cos(self.rotations.squeeze(-1))
        sin_t = torch.sin(self.rotations.squeeze(-1))
        s_h, s_w = self.scales[:, 0], self.scales[:, 1]

        sigma_hh = (s_h * cos_t) ** 2 + (s_w * sin_t) ** 2
        sigma_ww = (s_h * sin_t) ** 2 + (s_w * cos_t) ** 2
        radius_h = self.truncation_sigma * torch.sqrt(sigma_hh)
        radius_w = self.truncation_sigma * torch.sqrt(sigma_ww)

        min_h = self.means[:, 0] - radius_h
        max_h = self.means[:, 0] + radius_h
        min_w = self.means[:, 1] - radius_w
        max_w = self.means[:, 1] + radius_w

        rect_min_row = (min_h / self.tile_size).floor().long().clamp(0, self.grid_h - 1)
        rect_max_row = (max_h / self.tile_size).ceil().long().clamp(0, self.grid_h - 1)
        rect_min_col = (min_w / self.tile_size).floor().long().clamp(0, self.grid_w - 1)
        rect_max_col = (max_w / self.tile_size).ceil().long().clamp(0, self.grid_w - 1)

        rows = rect_max_row - rect_min_row + 1
        cols = rect_max_col - rect_min_col + 1
        valid_mask = (
            (rows <= self.max_tile_span) & (cols <= self.max_tile_span) &
            (rows > 0) & (cols > 0)
        )
        if valid_mask.sum() == 0:
            return

        rect_min_row    = rect_min_row[valid_mask]
        rect_max_row    = rect_max_row[valid_mask]
        rect_min_col    = rect_min_col[valid_mask]
        rect_max_col    = rect_max_col[valid_mask]
        rows            = rows[valid_mask]
        cols            = cols[valid_mask]
        original_indices = torch.where(valid_mask)[0]

        max_h_span = int(rows.max().item())
        max_w_span = int(cols.max().item())

        h_off = torch.arange(max_h_span, device=device)
        w_off = torch.arange(max_w_span, device=device)

        grid_rows = (rect_min_row[:, None, None] + h_off[None, :, None]
                     ).expand(-1, -1, max_w_span)
        grid_cols = (rect_min_col[:, None, None] + w_off[None, None, :]
                     ).expand(-1, max_h_span, -1)

        mask = (
            (grid_rows <= rect_max_row[:, None, None]) &
            (grid_cols <= rect_max_col[:, None, None])
        )
        valid_rows = grid_rows[mask]
        valid_cols = grid_cols[mask]
        valid_gids = original_indices[:, None, None].expand(
            -1, max_h_span, max_w_span)[mask]

        tile_ids = valid_rows * self.grid_w + valid_cols   # [M]

        # 计数排序：O(M + T)
        counts = torch.bincount(tile_ids, minlength=self.num_tiles)
        self.cell_ptr = torch.zeros(self.num_tiles + 1, dtype=torch.long, device=device)
        torch.cumsum(counts, dim=0, out=self.cell_ptr[1:])
        order = torch.argsort(tile_ids, stable=True)
        self.cell_indices = valid_gids[order]


# ─────────────────────────────────────────────────────────────────────────────
#  辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def _has_jacobian(model) -> bool:
    """检查模型是否启用了雅可比矩阵（J_params 非 None）。"""
    return getattr(model, 'enable_jacobian', False) and model.J_params is not None


def build_event_density_map(events, H, W, smooth_k=9, device='cuda'):
    """
    构建平滑事件密度图 [H, W]。
    events[:,0]=h（行坐标），events[:,1]=w（列坐标）。
    输出在指定 device 上。
    """
    events_map = torch.zeros(H, W, device=device)
    hs = events[:, 0].long().clamp(0, H - 1)
    ws = events[:, 1].long().clamp(0, W - 1)
    events_map.index_put_((hs, ws), torch.ones(len(events), device=device), accumulate=True)
    return F.avg_pool2d(events_map[None, None], smooth_k, stride=1,
                        padding=smooth_k // 2).squeeze()


def flow_regularizer(flow, eps=1e-6, div_q=0.8, tau_detach=True):
    u = flow.unsqueeze(0)  # [1, 2, H, W]
    u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')

    du_dh = 0.5 * (u_pad[:, 0:1, 2:,   1:-1] - u_pad[:, 0:1, :-2,  1:-1])
    dv_dw = 0.5 * (u_pad[:, 1:2, 1:-1, 2:  ] - u_pad[:, 1:2, 1:-1, :-2 ])

    div = du_dh + dv_dw
    abs_div = div.abs()

    tau = torch.quantile(abs_div.detach(), div_q)
    excess_div = torch.clamp(abs_div - tau, min=0.0)
    loss_div = torch.sqrt(excess_div ** 2 + eps ** 2).mean()

    d2u_dhh = u_pad[:, 0:1, 2:,   1:-1] - 2.0 * u_pad[:, 0:1, 1:-1, 1:-1] + u_pad[:, 0:1, :-2,  1:-1]
    d2u_dww = u_pad[:, 0:1, 1:-1, 2:  ] - 2.0 * u_pad[:, 0:1, 1:-1, 1:-1] + u_pad[:, 0:1, 1:-1, :-2 ]

    d2v_dhh = u_pad[:, 1:2, 2:,   1:-1] - 2.0 * u_pad[:, 1:2, 1:-1, 1:-1] + u_pad[:, 1:2, :-2,  1:-1]
    d2v_dww = u_pad[:, 1:2, 1:-1, 2:  ] - 2.0 * u_pad[:, 1:2, 1:-1, 1:-1] + u_pad[:, 1:2, 1:-1, :-2 ]

    loss_smooth = torch.sqrt(
        d2u_dhh**2 + d2u_dww**2 + d2v_dhh**2 + d2v_dww**2 + eps**2
    ).mean()

    return loss_smooth, loss_div

# ─────────────────────────────────────────────────────────────────────────────
#  GaussianVelocityField_Tiled
# ─────────────────────────────────────────────────────────────────────────────

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
            self.J_params = nn.Parameter(0.05 * torch.randn(num_gaussians, 3, device=device))
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
            a, b, c = self.J_params.unbind(-1)
            J_local = torch.stack([
                torch.stack([ a,  b], dim=-1),
                torch.stack([ c, -a], dim=-1)
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

        gauss_w = (torch.exp(-0.5 * term)
                   * torch.sigmoid(-2.0 * (term - 14.0)))
        gauss_w = gauss_w * valid_gmask.unsqueeze(1) * pixel_valid.unsqueeze(2)
        weights = gauss_w * cur_opacity.squeeze(-1).unsqueeze(1)   # [T, P, K]

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
        if device.type == 'cuda':
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
            a, b, c = self.J_params.unbind(-1)
            J_local = torch.stack([
                torch.stack([ a,  b], dim=-1),
                torch.stack([ c, -a], dim=-1)
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
        gauss_w = torch.exp(-0.5 * term) * torch.sigmoid(-2.0 * (term - 14.0))
        gauss_w = gauss_w * valid_gmask.unsqueeze(1) * pixel_valid.unsqueeze(2)
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


# ─────────────────────────────────────────────────────────────────────────────
#  优化器 & 阶段判断
# ─────────────────────────────────────────────────────────────────────────────

def create_optimizer(model, lr_mu, lr_j):
    param_groups = [
        {'params': [model.mu],            'lr': lr_mu, 'name': "mu"},
        {'params': [model.log_scale],     'lr': 0.05,  'name': "scale"},
        {'params': [model.rotation],      'lr': 0.05,  'name': "rotation"},
        {'params': [model.opacity_logit], 'lr': 0.1,   'name': "opacity"},
        {'params': [model.local_velocity],'lr': 0.05,  'name': "velocity"},
    ]
    if hasattr(model, 'J_params') and model.J_params is not None:
        param_groups.append({'params': [model.J_params], 'lr': lr_j, 'name': "jacobian"})
    return optim.Adam(param_groups, betas=(0.9, 0.999), eps=1e-15)


def decide_phase(step, phase):
    """
    Returns: set，包含 'clone' | 'split' | 'prune' 的任意组合
    """
    if step < phase[0]:
        return set()
    elif phase[0] <= step < phase[1]:
        return {'clone', 'prune', 'split'}
    elif phase[1] <= step <= phase[2]:
        return {'split', 'prune'}
    else:
        return {'prune'}


# ─────────────────────────────────────────────────────────────────────────────
#  优化器状态同步工具
# ─────────────────────────────────────────────────────────────────────────────
def replace_tensor_to_optimizer(optimizer, tensor_name, new_tensor):
    for group in optimizer.param_groups:
        if group.get("name") == tensor_name:
            old_param = group["params"][0]
            num_new   = new_tensor.shape[0] - old_param.shape[0]
            group["params"][0] = new_tensor
            if old_param in optimizer.state:
                old_state = optimizer.state.pop(old_param)
                new_state = {}
                for key, value in old_state.items():
                    if key in ['exp_avg', 'exp_avg_sq']:
                        zeros = torch.zeros(num_new, *value.shape[1:],
                                            device=value.device, dtype=value.dtype)
                        new_state[key] = torch.cat([value, zeros], dim=0)
                    elif key == 'step':
                        new_state[key] = value
                optimizer.state[new_tensor] = new_state
            return


def prune_tensor_in_optimizer(optimizer, tensor_name, new_tensor, keep_mask):
    for group in optimizer.param_groups:
        if group.get("name") == tensor_name:
            old_param = group["params"][0]
            group["params"][0] = new_tensor
            if old_param in optimizer.state:
                old_state = optimizer.state.pop(old_param)
                new_state = {}
                for key, value in old_state.items():
                    if key in ['exp_avg', 'exp_avg_sq']:
                        new_state[key] = value[keep_mask]
                    elif key == 'step':
                        new_state[key] = value
                optimizer.state[new_tensor] = new_state
            return


_ATTR_MAPPING = {
    'mu':       'mu',
    'scale':    'log_scale',
    'rotation': 'rotation',
    'opacity':  'opacity_logit',
    'velocity': 'local_velocity',
    'jacobian': 'J_params',
}


def append_gaussians(model, optimizer, extension_dict):
    for name, new_data in extension_dict.items():
        if new_data is None:
            continue
        if name == 'jacobian' and not _has_jacobian(model):
            continue
        attr_name = _ATTR_MAPPING.get(name)
        if attr_name is None or not hasattr(model, attr_name):
            continue
        old_param = getattr(model, attr_name)
        if old_param is None:
            continue
        new_param = nn.Parameter(torch.cat([old_param, new_data], dim=0))
        setattr(model, attr_name, new_param)
        replace_tensor_to_optimizer(optimizer, name, new_param)
    model.num_gaussians = model.mu.shape[0]


def prune_gaussians(model, optimizer, mask_to_remove):
    keep_mask = ~mask_to_remove
    for name, attr_name in _ATTR_MAPPING.items():
        if name == 'jacobian' and not _has_jacobian(model):
            continue
        if not hasattr(model, attr_name):
            continue
        old_param = getattr(model, attr_name)
        if old_param is None:
            continue
        new_param = nn.Parameter(old_param[keep_mask])
        setattr(model, attr_name, new_param)
        prune_tensor_in_optimizer(optimizer, name, new_param, keep_mask)
    model.num_gaussians = model.mu.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
#  密度控制辅助
# ─────────────────────────────────────────────────────────────────────────────

def precompute_clone_candidates(events_map, quantile=0.3):
    threshold = events_map.quantile(quantile)
    return torch.where(((events_map < threshold) & (events_map > 0)).reshape(-1))[0]


def sample_maps_at_gaussians(maps, mu, H, W):
    N      = mu.shape[0]
    device = mu.device
    mu_norm       = torch.empty_like(mu)
    mu_norm[:, 0] = mu[:, 0] / (H - 1) * 2 - 1
    mu_norm[:, 1] = mu[:, 1] / (W - 1) * 2 - 1
    grid    = mu_norm[:, [1, 0]].view(1, 1, N, 2)
    stacked = torch.stack(maps, dim=0).unsqueeze(0)
    sampled = F.grid_sample(
        stacked, grid, mode='bilinear', align_corners=True, padding_mode='border'
    ).squeeze(0).squeeze(1).T
    return [sampled[:, i] for i in range(len(maps))]

def compute_flow_curvature(pred_flow):
    flow4   = pred_flow.unsqueeze(0)
    sobel_w = _SOBEL_W.to(pred_flow.device)   # ∂/∂w
    sobel_h = _SOBEL_H.to(pred_flow.device)   # ∂/∂h
    gw = F.conv2d(flow4, sobel_w, padding=1, groups=2)
    gh = F.conv2d(flow4, sobel_h, padding=1, groups=2)
    grad_mean = (gw ** 2 + gh ** 2).sum(dim=1).sqrt().mean()

    lap_ww = flow4[:, :, :, 2:] - 2 * flow4[:, :, :, 1:-1] + flow4[:, :, :, :-2]
    lap_hh = flow4[:, :, 2:, :] - 2 * flow4[:, :, 1:-1, :] + flow4[:, :, :-2, :]
    lap_ww = F.pad(lap_ww, (1, 1, 0, 0))
    lap_hh = F.pad(lap_hh, (0, 0, 1, 1))
    curv_map  = (lap_ww ** 2 + lap_hh ** 2).sum(dim=1).sqrt().squeeze(0)
    curv_mean = curv_map.mean()
    
    return curv_map


def adaptive_density_control(
    model,
    optimizer,
    pred_flow,
    weight_map,
    events_map,
    clone_candidates,
    phase,
    max_clone             = 50,
    max_split             = 50,
    prune_opacity_thresh  = 0.1,
    max_anisotropy        = 3.0,
    verbose               = True,
):
    device = model.mu.device
    H, W   = model.H, model.W
    margin = model.tile_size // 2
    use_J  = _has_jacobian(model)

    with torch.no_grad():
        mu        = model.mu.detach()
        opacity   = torch.sigmoid(model.opacity_logit).squeeze(-1)
        scales    = torch.exp(model.log_scale)
        max_scale = scales.max(dim=1).values
        min_scale = scales.min(dim=1).values.clamp(min=1e-6)
        v_norm    = model.local_velocity.detach().norm(dim=1)
        N         = mu.shape[0]

        flow_curvature = compute_flow_curvature(pred_flow)

        # ══════ Clone ══════
        n_clone    = 0
        clone_dict = None

        if 'clone' in phase and clone_candidates.shape[0] > 0:
            n_sample     = min(max_clone, clone_candidates.shape[0])
            perm         = torch.randperm(clone_candidates.shape[0], device=device)[:n_sample]
            sampled_flat = clone_candidates[perm]
            clone_y      = sampled_flat // W
            clone_x      = sampled_flat  % W

            clone_pos = torch.stack([clone_y.float(), clone_x.float()], dim=1)
            dist_to_existing = torch.cdist(clone_pos, mu)
            keep      = dist_to_existing.min(dim=1).values > (model.tile_size * 0.2)
            clone_pos = clone_pos[keep]
            n_clone   = clone_pos.shape[0]

            if n_clone > 0:
                nearest  = dist_to_existing[keep].argmin(dim=1)
                mu_new   = clone_pos + torch.randn_like(clone_pos) * 2.0
                clone_dict = {
                    'mu':       mu_new,
                    'scale':    model.log_scale.data[nearest].clone(),
                    'rotation': model.rotation.data[nearest].clone(),
                    'opacity':  torch.zeros(n_clone, 1, device=device),
                    'velocity': model.local_velocity.data[nearest].clone(),
                    'jacobian': model.J_params.data[nearest].clone() if use_J else None,
                }

        # ══════ Split ══════
        n_split    = 0
        split_dict = None
        # 以下变量仅在 split 分支内赋值，verbose 打印守卫在同一分支内
        curv_thresh       = None
        mask_high_curv    = None
        mask_lowvel_large = None

        if 'split' in phase:
            flow_curvature_smooth = F.avg_pool2d(
                flow_curvature[None, None], 5, stride=1, padding=2
            ).squeeze()
            flow_mag = pred_flow.norm(dim=0)

            curvature_at_mu, flowmag_at_mu = sample_maps_at_gaussians(
                [flow_curvature_smooth, flow_mag], mu, H, W
            )

            large_gaussian = max_scale > max_scale.quantile(0.8)

            mask_high_curv    = (curvature_at_mu > curvature_at_mu.quantile(0.8)) & large_gaussian
            mask_lowvel_large = (
                (flowmag_at_mu < flowmag_at_mu.median()) & large_gaussian
            )
            mask_split    = mask_high_curv | mask_lowvel_large
            split_indices = torch.where(mask_split)[0]

            if split_indices.shape[0] > max_split:
                topk          = torch.topk(curvature_at_mu[split_indices], max_split).indices
                split_indices = split_indices[topk]

            n_split = split_indices.shape[0]

            if n_split > 0:
                mu_s            = model.mu.data[split_indices].clone()
                r               = model.rotation.data[split_indices].squeeze(-1)
                axis            = torch.stack([torch.cos(r), torch.sin(r)], dim=-1)
                child_log_scale = model.log_scale.data[split_indices] - 0.5 * np.log(2.0)
                child_scale     = torch.exp(child_log_scale)
                child_max_s     = child_scale.max(dim=1, keepdim=True).values
                
                # 用更保守一点的偏移，避免一下子分太开
                offset = axis * (0.5 * child_max_s)
                
                mu_center = model.mu.data[split_indices].clone()
                
                # parent 也缩小
                model.mu.data[split_indices] = mu_center + offset
                model.log_scale.data[split_indices] = child_log_scale.clone()
                
                split_dict = {
                    'mu':       mu_center - offset,
                    'scale':    child_log_scale.clone(),
                    'rotation': model.rotation.data[split_indices].clone(),
                    'opacity':  model.opacity_logit.data[split_indices].clone(),
                    'velocity': model.local_velocity.data[split_indices].clone(),
                    'jacobian': model.J_params.data[split_indices].clone() if use_J else None,
                }

        # ══════ Prune mask ══════
        mask_geometry = (
            (max_scale < 0.5)                               |
            (max_scale > H / 6)                             |
            ((max_scale / min_scale) > max_anisotropy)      |
            (mu[:, 0] < -margin) | (mu[:, 0] > H + margin) |
            (mu[:, 1] < -margin) | (mu[:, 1] > W + margin)
        )

        mask_prune = (opacity < prune_opacity_thresh) | mask_geometry

    if verbose:
        if 'clone' in phase:
            print(f"  [Clone] candidates={clone_candidates.shape[0]}  after_filter={n_clone}")
        if 'split' in phase:
            curv_info = (f"curv_thresh={curv_thresh:.4f}  "
                         f"high_curv={mask_high_curv.sum().item()}  "
                         f"low_vel_large={mask_lowvel_large.sum().item()}"
                         if curv_thresh is not None else "curv=zero(fallback)")
            print(f"total_triggered={n_split}  ")
        if 'prune' in phase:
            print(f"  [Prune] opacity_mean={opacity.mean():.3f}  "
                  f"geometry={mask_geometry.sum().item()}  "
                  f"total={mask_prune.sum().item()}")

    # ══════ 执行：Clone → Split → Prune ══════
    has_changed = False

    if n_clone > 0 and clone_dict is not None:
        append_gaussians(model, optimizer, clone_dict)
        has_changed = True

    if n_split > 0 and split_dict is not None:
        append_gaussians(model, optimizer, split_dict)
        
        has_changed = True

    n_pruned = 0
    if 'prune' in phase:
        current_N  = model.num_gaussians
        prune_full = torch.zeros(current_N, dtype=torch.bool, device=device)
        prune_full[:N] = mask_prune

        if prune_full.any():
            n_pruned       = prune_full.sum().item()
        
            prune_gaussians(model, optimizer, prune_full)
            has_changed    = True

    if verbose:
        print(f"  [{phase}] Clone:{n_clone}  Split:{n_split}  "
              f"Prune:{n_pruned}  -> N={model.num_gaussians}")

    return has_changed


# ─────────────────────────────────────────────────────────────────────────────
#  训练历史可视化
# ─────────────────────────────────────────────────────────────────────────────

def _plot_training_history(history, phase):
    steps        = range(len(history['total']))
    phase_colors = ['#f0e68c', '#c8e6c9', '#bbdefb']
    phase_labels = ['Phase 0\n(warm-up)', 'Phase 1\n(clone+split+prune)',
                    'Phase 2\n(split+prune)', 'Phase 3\n(prune only)']
    boundaries   = [0] + phase + [len(history['total'])]

    fig, axes = plt.subplots(4, 1, figsize=(6, 5), sharex=True)
    fig.suptitle('Training History', fontsize=14, fontweight='bold')

    panels = [
        ('total',       axes[0], 'Total Loss',    '#e74c3c'),
        ('data',        axes[0], 'Data Loss',      '#3498db'),
        ('div',         axes[1], 'Div Loss (×10)', '#9b59b6'),
        ('smooth',      axes[1], 'Smooth Loss',    '#1abc9c'),
        ('J_grad_norm', axes[2], 'J Grad Norm',    '#e67e22'),
        ('N',           axes[3], '# Gaussians',    '#2c3e50'),
    ]

    for ax in axes:
        for k, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            ax.axvspan(s, e, alpha=0.08,
                       color=phase_colors[min(k, len(phase_colors) - 1)])
        for b in phase:
            ax.axvline(b, color='gray', linewidth=0.8, linestyle='--')

    for key, ax, label, color in panels:
        vals = history[key]
        if key == 'div':
            vals = [v * 10 for v in vals]
        ax.plot(steps, vals, label=label, color=color, linewidth=1.2, alpha=0.9)

    axes[0].set_ylabel('Loss');       axes[0].legend(loc='upper right', fontsize=8)
    axes[1].set_ylabel('Reg Loss');   axes[1].set_yscale('log'); axes[1].legend(loc='upper right', fontsize=8)
    axes[2].set_ylabel('Grad Norm');  axes[2].legend(loc='upper right', fontsize=8)
    axes[3].set_ylabel('N');          axes[3].legend(loc='upper right', fontsize=8)
    axes[3].set_xlabel('Iteration')

    # draw first so y-axis limits are finalised before placing text labels
    plt.tight_layout()
    fig.canvas.draw()
    for k, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ylim_top = axes[0].get_ylim()[1]
        axes[0].text((s + e) / 2, ylim_top,
                     phase_labels[k], ha='center', va='bottom',
                     fontsize=7, color='gray')

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Jacobian lr 自动选择
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_lr_j(events_map, cfg: GaussianFlowConfig,
                  supervised: bool) -> float:
    """
    Return the initial Jacobian learning rate.

    Priority:
    1. cfg.lr_j is not None  -> use it directly (manual override).
    2. supervised=True       -> use 0.001 (flow type already known, skip detection).
    3. supervised=False      -> auto-detect from events:
           turbulence     -> 0.03
           non-turbulence -> 0.001
    lr stays constant throughout training (no scheduler decay on jacobian).
    """
    if cfg.lr_j is not None:
        if cfg.verbose:
            print(f"[AutoLR] lr_jacobian = {cfg.lr_j} (manual)")
        return cfg.lr_j

    if supervised:
        if cfg.verbose:
            print("[AutoLR] supervised mode  ->  lr_jacobian = 0.001")
        return 0.001

    turb = is_turbulence(events_map)
    lr_j = 0.05 if turb else 0.001
    if cfg.verbose:
        tag = "turbulence" if turb else "non-turbulence"
        print(f"[AutoLR] flow detected: {tag}  ->  lr_jacobian = {lr_j} (auto)")
    return lr_j

def get_rebuild_interval(i):
    if i < 30:
        return 1
    elif i < 80:
        return 5
    else:
        return 10
        
# ─────────────────────────────────────────────────────────────────────────────
#  公开接口（三层）
#
#  train_gaussian_flow  ── 只负责训练，返回 (model, history)
#  render_gaussian_flow ── 只负责渲染，接受已训练的 model，支持任意分辨率
#  run_gaussian_flow    ── 便捷封装，依次调用上面两个，一次返回全部结果
# ─────────────────────────────────────────────────────────────────────────────


def train_gaussian_flow(
    events: torch.Tensor,
    cfg: GaussianFlowConfig,
    flowgt: Optional[torch.Tensor] = None,
) -> tuple:
    """
    训练高斯速度场，返回训练完毕的模型和损失历史。

    Parameters
    ----------
    events : torch.Tensor, shape (N, 4)
        事件数据 [h, w, t, p]。
    cfg : GaussianFlowConfig
        所有超参数。
    flowgt : torch.Tensor or None, shape (2, H, W)
        有监督时传入 ground-truth flow，无监督传 None。

    Returns
    -------
    model : GaussianVelocityField_Tiled
        训练完成的模型，保持在 cfg.device 上，处于 train() 状态。
        高斯参数定义在 [0, cfg.H) × [0, cfg.W) 坐标空间中。
    history : dict
        训练过程各项损失及统计量
        键：'total', 'data', 'div', 'smooth', 'J_grad_norm', 'N'

    Notes
    -----
    训练期间以 compute_derivatives=False 调用 model.forward()，
    不计算解析散度/旋度，减少不必要的内存和计算开销。
    训练结束后模型保持 train() 状态，需要渲染时调用 render_gaussian_flow()。
    """
    device     = cfg.device
    supervised = flowgt is not None

    events = events.to(device)
    if flowgt is not None:
        flowgt = flowgt.to(device)

    events_map = build_event_density_map(
        events, cfg.H, cfg.W, smooth_k=cfg.density_smooth_k, device=device)
    clone_candidates = precompute_clone_candidates(
        events_map, quantile=cfg.clone_candidates_quantile)

    model = GaussianVelocityField_Tiled(
        num_gaussians   = cfg.init_n,
        H               = cfg.H,
        W               = cfg.W,
        tile_size       = cfg.tile_size,
        events_map      = events_map,
        density_alpha   = cfg.density_alpha,
        enable_jacobian = cfg.enable_jacobian,
        device          = device,
    ).to(device)
    model.rebuild_indexer()

    lr_j = _resolve_lr_j(events_map, cfg, supervised)
    optimizer = create_optimizer(model, lr_mu=cfg.lr_mu, lr_j=lr_j)

    lr_lambda_mu = lambda step: (1 / 5) ** (step / cfg.iterations)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=[
            lr_lambda_mu if g["name"] == "mu" else (lambda s: 1.0)
            for g in optimizer.param_groups
        ]
    )

    cmax_loss_fn = ContrastMaximizationLoss(
        loss_type   = cfg.cmax_loss_type,
        sigma       = cfg.cmax_sigma,
        kernel_size = cfg.cmax_kernel_size,
    ).to(device)

    if cfg.verbose:
        print(f"开始训练 | 初始高斯数: {model.num_gaussians}")

    history        = defaultdict(list)

    for i in range(cfg.iterations):
        optimizer.zero_grad(set_to_none=True)
        
        # 训练期间只需 pred_flow 用于梯度计算，不计算解析散度/旋度
        pred_flow, weight_map, _, _ = model(
            rebuild_index=(i == 0), compute_derivatives=False)

        if supervised:
            loss_data = F.mse_loss(pred_flow, flowgt)
        else:
            loss_data = cmax_loss_fn(events, pred_flow, (cfg.H, cfg.W))

        smooth_loss, div_loss = flow_regularizer(pred_flow)
        total_loss             = loss_data + cfg.w_div * div_loss + cfg.w_smooth * smooth_loss

        total_loss.backward()

        j_grad_norm = 0.0
        if model.J_params is not None and model.J_params.grad is not None:
            j_grad_norm = model.J_params.grad.norm().item()

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.log_scale.clamp_(min=cfg.log_scale_min, max=np.log(cfg.H / 6))

        history['total'].append(total_loss.item())
        history['data'].append(loss_data.item())
        history['div'].append(div_loss.item())
        history['smooth'].append(smooth_loss.item())
        history['J_grad_norm'].append(j_grad_norm)
        history['N'].append(model.num_gaussians)

        rebuild_interval_now = get_rebuild_interval(i)
        if i % rebuild_interval_now == 0:
            model.rebuild_indexer()

        if i >= cfg.phase[0] and i % cfg.density_interval == 0:
            phase_flag = decide_phase(i, cfg.phase)
            if phase_flag:
                has_changed = adaptive_density_control(
                    model            = model,
                    optimizer        = optimizer,
                    pred_flow        = pred_flow.detach(),
                    weight_map       = weight_map.detach(),
                    events_map       = events_map,
                    clone_candidates = clone_candidates,
                    phase            = phase_flag,
                    max_clone             = cfg.max_clone,
                    max_split             = cfg.max_split,
                    prune_opacity_thresh     = cfg.prune_opacity_thresh,
                    max_anisotropy        = cfg.max_anisotropy,
                    verbose               = cfg.verbose,
                )
                if has_changed:
                    model.rebuild_indexer()

        if (i % 50 == 0 or i == (cfg.iterations-1))and cfg.verbose:
            print(f"Iter {i:04d} | total={total_loss.item():.6f} | "
                  f"data={loss_data.item():.6f} | "
                  f"div={div_loss.item():.6f} | "
                  f"smooth={smooth_loss.item():.6f} | "
                  f"J_grad={j_grad_norm:.5f} | "
                  f"N={model.num_gaussians}")

    if cfg.verbose:
        print("训练结束")
        _plot_training_history(history, cfg.phase)

    return model, dict(history)


def render_gaussian_flow(
    model,
    eval_H: Optional[int] = None,
    eval_W: Optional[int] = None,
    verbose: bool = True,
) -> GaussianFlowResult:
    """
    对已训练的高斯速度场模型进行验证推理，输出所有物理量。

    高斯参数保持不变，仅通过改变查询网格密度实现任意分辨率输出。
    可对同一个训练好的模型多次调用，以不同分辨率获取结果，无需重复训练。

    Parameters
    ----------
    model : GaussianVelocityField_Tiled
        由 train_gaussian_flow() 返回的已训练模型。
    eval_H : int or None
        输出行分辨率。None 时使用模型训练分辨率 model.H。
    eval_W : int or None
        输出列分辨率。None 时使用模型训练分辨率 model.W。
    verbose : bool
        是否打印推理进度。

    Returns
    -------
    GaussianFlowResult
        .flow       : np.ndarray (2, eval_H, eval_W)，通道0=vel_h，通道1=vel_w
        .model      : 传入的 model（引用，未修改）
        .weight_map : torch.Tensor (1, eval_H, eval_W)
        .div        : torch.Tensor (eval_H, eval_W)，解析散度 ∂vel_h/∂h + ∂vel_w/∂w
        .vorticity  : torch.Tensor (eval_H, eval_W)，解析旋度 ∂vel_w/∂h − ∂vel_h/∂w
        .eval_H     : int，实际输出行分辨率
        .eval_W     : int，实际输出列分辨率
        .history    : dict，空字典（历史由 train_gaussian_flow 持有）

    Notes
    -----
    推理期间自动切换 model.eval() + torch.no_grad()，结束后恢复 model.train()。
    render_at_resolution() 在高斯坐标空间中直接以分数坐标查询，无插值损失。

    Examples
    --------
    # 一次训练，多次渲染不同分辨率
    model, history = train_gaussian_flow(events, cfg)

    result_256  = render_gaussian_flow(model, 256, 256)
    result_512  = render_gaussian_flow(model, 512, 512)
    result_1024 = render_gaussian_flow(model, 1024, 1024)
    """
    eval_H = eval_H if eval_H is not None else model.H
    eval_W = eval_W if eval_W is not None else model.W

    if verbose:
        print(f"验证推理中（训练分辨率 {model.H}×{model.W}"
              f" → 输出分辨率 {eval_H}×{eval_W}）...")

    model.eval()
    with torch.no_grad():
        pred_flow, weight_map, div, vorticity = model.render_at_resolution(
            eval_H, eval_W, compute_derivatives=True)

        flow_np    = pred_flow.cpu().numpy()
        weight_cpu = weight_map.cpu()
        div_cpu    = div.cpu()        if div       is not None else torch.zeros(eval_H, eval_W)
        vort_cpu   = vorticity.cpu() if vorticity is not None else torch.zeros(eval_H, eval_W)

    model.train()

    if verbose:
        print("验证推理完成")

    return GaussianFlowResult(
        flow       = flow_np,
        model      = model,
        weight_map = weight_cpu,
        div        = div_cpu,
        vorticity  = vort_cpu,
        eval_H     = eval_H,
        eval_W     = eval_W,
        history    = {},
    )


def run_gaussian_flow(
    events: torch.Tensor,
    cfg: GaussianFlowConfig,
    flowgt: Optional[torch.Tensor] = None,
    eval_H: Optional[int] = None,
    eval_W: Optional[int] = None,
) -> GaussianFlowResult:
    """
    便捷封装：依次调用 train_gaussian_flow() 和 render_gaussian_flow()。

    适合只需要一次性结果的场景。若需要对同一模型渲染多种分辨率，
    请直接调用 train_gaussian_flow() + render_gaussian_flow()。

    Parameters
    ----------
    events  : torch.Tensor, shape (N, 4)
    cfg     : GaussianFlowConfig
    flowgt  : torch.Tensor or None, shape (2, H, W)
    eval_H  : int or None，输出行分辨率，None 时使用 cfg.H
    eval_W  : int or None，输出列分辨率，None 时使用 cfg.W

    Returns
    -------
    GaussianFlowResult
        同 render_gaussian_flow()，另 .history 包含完整训练历史。
    """
    model, history = train_gaussian_flow(events, cfg, flowgt)
    result = render_gaussian_flow(model, eval_H, eval_W, verbose=cfg.verbose)
    result.history = history   # 将训练历史挂回结果
    return result