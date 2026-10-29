import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional


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
    """所有超参数集中在此处。"""

    # ── 分辨率 ──────────────────────────────────────────────────────────────
    H: int = 256                    # 流场分辨率（行）
    W: int = 256                    # 流场分辨率（列）

    # ── 模型初始化 ───────────────────────────────────────────────────────────
    init_n: int = 300               # 初始高斯数量
    tile_size: int = 32             # 分块大小
    init_scale_h: float = 7.0       # 实际尺度 s_h 独立均匀采样于 [1, init_scale_h)，再取 log
    init_scale_w: float = 7.0       # 实际尺度 s_w 独立均匀采样于 [1, init_scale_w)，再取 log
    enable_jacobian: bool = True    # 是否启用雅可比矩阵参数
    jacobian_mode: str = "full"     # full=[[a,b],[c,d]]；trace_free=[[a,b],[c,-a]]

    # ── 训练控制 ─────────────────────────────────────────────────────────────
    iterations: int = 300           # 总迭代次数
    rebuild_interval: int = 10      # 重建空间索引的间隔
    enable_density_control: bool = True  # False 时固定 init_n，跳过 clone/split/prune
    density_interval: int = 50      # 密度控制（clone/split/prune）执行间隔
    phase: List[int] = field(default_factory=lambda: [50, 200, 250])   # 训练阶段分界点 [phase0_end, phase1_end, phase2_end]

    # ── 优化器 lr ────────────────────────────────────────────────────────────
    lr_mu: float = 3.0              # 均值参数初始学习率；线性衰减到 20% （默认 3.0 → 0.6）
    lr_j: Optional[float] = None   # Jacobian lr；None = 自动检测（湍流→0.05，其他→0.001）

    # ── 损失权重 ─────────────────────────────────────────────────────────────
    w_div: float = 15             # 散度正则化权重
    div_top_ratio: float = 0.1    # 仅约束 |div| 最大的比例；0=关闭，1=全场
    div_last: int = 500

    # ── log_scale clamp 范围 ─────────────────────────────────────────────────
    log_scale_min: float = np.log(0.5)

    # ── 密度控制参数 ─────────────────────────────────────────────────────────
    max_clone: int = 50
    max_split: int = 50
    prune_opacity_thresh: float = 0.1
    max_anisotropy: float = 3.0
    residual_block_size: int = 16
    residual_min_events_per_block: int = 0
    clone_residual_quantile: float = 0.3
    clone_coverage_quantile: float = 0.3
    split_residual_quantile: float = 0.5
    split_curvature_quantile: float = 0.5
    split_scale_quantile: float = 0.7
    split_scale_shrink: float = 0.7071   # 1 / sqrt(2)
    split_offset_factor: float = 0.5
    split_support_samples: int = 5
    split_support_extent: float = 1.5
    split_fallback_topk: int = 8

    # ── 事件密度图平滑核 ─────────────────────────────────────────────────────
    density_smooth_k: int = 9

    # ── Cmax 损失参数 ────────────────────────────────────────────────────────
    cmax_loss_type: str = 'l2_max'
    cmax_sigma: float = 0.6
    cmax_kernel_size: int = 5

    # ── 设备 ────────────────────────────────────────────────────────────────
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    verbose: bool = False
