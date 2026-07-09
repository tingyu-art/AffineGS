import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Optional

from CMax import ContrastMaximizationLoss
from utils.flow_classifier import is_turbulence

from .config import GaussianFlowConfig, GaussianFlowResult
from .density_control import adaptive_density_control, create_optimizer, decide_phase
from .model import GaussianVelocityField_Tiled
from .utils import build_event_density_map, divergence_regularizer


def _plot_training_history(history, phase):
    steps        = range(len(history['total']))
    phase_colors = ['#f0e68c', '#c8e6c9', '#bbdefb']
    phase_labels = ['Phase 0\n(warm-up)', 'Phase 1\n(clone+split+prune)',
                    'Phase 2\n(split+prune)', 'Phase 3\n(prune only)']
    boundaries   = [0] + phase + [len(history['total'])]

    fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
    fig.suptitle('Training History', fontsize=14, fontweight='bold')

    panels = [
        ('total',       axes[0], 'Total Loss',    '#e74c3c'),
        ('data',        axes[0], 'Data Loss',      '#3498db'),
        ('div',         axes[1], 'Div Loss (×10)', '#9b59b6'),
        ('N',           axes[2], '# Gaussians',    '#2c3e50'),
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
    axes[1].set_ylabel('Div Loss');   axes[1].set_yscale('log'); axes[1].legend(loc='upper right', fontsize=8)
    axes[2].set_ylabel('N');          axes[2].legend(loc='upper right', fontsize=8)
    axes[2].set_xlabel('Iteration')

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
           turbulence     -> 0.05
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
        键：'total', 'data', 'div', 'N'

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

    model = GaussianVelocityField_Tiled(
        num_gaussians   = cfg.init_n,
        H               = cfg.H,
        W               = cfg.W,
        tile_size       = cfg.tile_size,
        enable_jacobian = cfg.enable_jacobian,
        device          = device,
        jacobian_mode   = cfg.jacobian_mode,
        init_scale_h    = cfg.init_scale_h,
        init_scale_w    = cfg.init_scale_w,
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

        if i < cfg.div_last:
            weight_div = cfg.w_div
        else:
            weight_div = 0
            
        div_loss = divergence_regularizer(pred_flow, top_ratio=cfg.div_top_ratio)
        total_loss = (
            loss_data
            + weight_div * div_loss
        )

        total_loss.backward()

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            model.log_scale.clamp_(min=cfg.log_scale_min, max=np.log(cfg.H / 6))

        history['total'].append(total_loss.item())
        history['data'].append(loss_data.item())
        history['div'].append(div_loss.item())
        history['N'].append(model.num_gaussians)

        rebuild_interval_now = get_rebuild_interval(i)
        if i % rebuild_interval_now == 0:
            model.rebuild_indexer()

        if i >= cfg.phase[0] and i % cfg.density_interval == 0:
            phase_flag = decide_phase(i, cfg.phase)
            if phase_flag:
                has_changed = adaptive_density_control(
                    model                    = model,
                    optimizer                = optimizer,
                    pred_flow                = pred_flow.detach(),
                    weight_map               = weight_map.detach(),
                    events                   = events,
                    cmax_loss_fn             = cmax_loss_fn,
                    phase                    = phase_flag,
                    max_clone                = cfg.max_clone,
                    max_split                = cfg.max_split,
                    prune_opacity_thresh     = cfg.prune_opacity_thresh,
                    max_anisotropy           = cfg.max_anisotropy,
                    residual_block_size      = cfg.residual_block_size,
                    residual_min_events_per_block = cfg.residual_min_events_per_block,
                    clone_residual_quantile  = cfg.clone_residual_quantile,
                    clone_coverage_quantile  = cfg.clone_coverage_quantile,
                    split_residual_quantile  = cfg.split_residual_quantile,
                    split_curvature_quantile = cfg.split_curvature_quantile,
                    split_scale_quantile     = cfg.split_scale_quantile,
                    split_scale_shrink       = cfg.split_scale_shrink,
                    split_offset_factor      = cfg.split_offset_factor,
                    split_support_samples    = cfg.split_support_samples,
                    split_support_extent     = cfg.split_support_extent,
                    split_fallback_topk      = cfg.split_fallback_topk,
                    verbose                  = cfg.verbose,
                )
                if has_changed:
                    model.rebuild_indexer()

        if (i % 50 == 0 or i == (cfg.iterations-1))and cfg.verbose:
            print(f"Iter {i:04d} | total={total_loss.item():.6f} | "
                  f"data={loss_data.item():.6f} | "
                  f"div={div_loss.item():.6f} | "
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
