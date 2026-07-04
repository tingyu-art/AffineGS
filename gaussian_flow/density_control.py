import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .utils import _SOBEL_H, _SOBEL_W, _has_jacobian


def create_optimizer(model, lr_mu, lr_j):
    param_groups = [
        {'params': [model.mu],            'lr': lr_mu, 'name': "mu"},
        {'params': [model.log_scale],     'lr': 0.05,  'name': "scale"},
        {'params': [model.rotation],      'lr': 0.05,  'name': "rotation"},
        {'params': [model.opacity_logit], 'lr': 0.05,   'name': "opacity"},
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
    sobel_w = _SOBEL_W.to(pred_flow.device)
    sobel_h = _SOBEL_H.to(pred_flow.device)
    gw = F.conv2d(flow4, sobel_w, padding=1, groups=2)
    gh = F.conv2d(flow4, sobel_h, padding=1, groups=2)

    grad_map  = (gw ** 2 + gh ** 2).sum(dim=1).sqrt().squeeze(0)
    return  grad_map


def block_reduce_map(x, block_size, reduction='mean'):
    """将 [H, W] 图按 block_size×block_size 分块聚合为 [Hb, Wb]。"""
    H, W = x.shape
    Hb = H // block_size
    Wb = W // block_size
    if Hb == 0 or Wb == 0:
        raise ValueError(f"block_size={block_size} 过大，图像尺寸为 {H}x{W}")
    x = x[:Hb * block_size, :Wb * block_size]
    x = x.view(Hb, block_size, Wb, block_size).permute(0, 2, 1, 3)
    if reduction == 'mean':
        return x.mean(dim=(2, 3))
    if reduction == 'max':
        return x.amax(dim=(2, 3))
    raise ValueError(f"Unsupported reduction: {reduction}")


def robust_normalize(x, valid_mask=None, q_low=0.05, q_high=0.95, eps=1e-8):
    vals = x[valid_mask] if valid_mask is not None else x.reshape(-1)
    if vals.numel() == 0:
        return torch.zeros_like(x)
    lo = torch.quantile(vals, q_low)
    hi = torch.quantile(vals, q_high)
    if (hi - lo).abs() < eps:
        hi = lo + eps
    return ((x - lo) / (hi - lo + eps)).clamp(0.0, 1.0)


def sample_block_map_at_gaussians(block_map, mu, block_size, H, W):
    Hb, Wb = block_map.shape
    bh = (mu[:, 0] / block_size).floor().long().clamp(0, Hb - 1)
    bw = (mu[:, 1] / block_size).floor().long().clamp(0, Wb - 1)
    return block_map[bh, bw]


def aggregate_block_map_over_gaussian_support(
    block_map,
    mu,
    scales,
    rotation,
    block_size,
    sample_steps=5,
    support_extent=1.5,
):
    """
    在每个高斯的局部支撑域内采样 block_map，而不是只看高斯中心所在块。
    返回：
        weighted_mean : [N]
        support_max   : [N]
        support_mix   : [N] = 0.7 * mean + 0.3 * max
    """
    device = mu.device
    dtype = mu.dtype
    Hb, Wb = block_map.shape

    offs = torch.linspace(-support_extent, support_extent, sample_steps, device=device, dtype=dtype)
    oh, ow = torch.meshgrid(offs, offs, indexing='ij')
    local = torch.stack([oh.reshape(-1), ow.reshape(-1)], dim=-1)  # [S, 2], sigma units
    w = torch.exp(-0.5 * (local[:, 0] ** 2 + local[:, 1] ** 2))
    w = w / (w.sum() + 1e-8)

    sx = scales[:, 0:1]
    sy = scales[:, 1:2]
    r = rotation.squeeze(-1)
    cos_r = torch.cos(r).unsqueeze(1)
    sin_r = torch.sin(r).unsqueeze(1)

    local_h = local[:, 0].unsqueeze(0) * sx
    local_w = local[:, 1].unsqueeze(0) * sy

    off_h = cos_r * local_h - sin_r * local_w
    off_w = sin_r * local_h + cos_r * local_w

    sample_h = mu[:, 0:1] + off_h
    sample_w = mu[:, 1:2] + off_w

    bh = (sample_h / block_size).floor().long().clamp(0, Hb - 1)
    bw = (sample_w / block_size).floor().long().clamp(0, Wb - 1)

    vals = block_map[bh, bw]  # [N, S]
    weighted_mean = (vals * w.unsqueeze(0)).sum(dim=1)
    support_max = vals.max(dim=1).values
    support_mix = 0.7 * weighted_mean + 0.3 * support_max
    return weighted_mean, support_max, support_mix


def block_indices_to_centers(flat_idx, block_map_shape, block_size, device):
    Hb, Wb = block_map_shape
    bh = flat_idx // Wb
    bw = flat_idx % Wb
    centers_h = (bh.float() + 0.5) * block_size
    centers_w = (bw.float() + 0.5) * block_size
    return torch.stack([centers_h, centers_w], dim=1).to(device)


def adaptive_density_control(
    model,
    optimizer,
    pred_flow,
    weight_map,
    events,
    cmax_loss_fn,
    phase,
    max_clone,
    max_split,
    prune_opacity_thresh,
    max_anisotropy,
    residual_block_size,
    residual_min_events_per_block,
    clone_residual_quantile,
    clone_coverage_quantile,
    split_residual_quantile,
    split_curvature_quantile,
    split_scale_quantile,
    split_scale_shrink,
    split_offset_factor,
    split_support_samples,
    split_support_extent,
    split_fallback_topk,
    verbose=True,
):
    device = model.mu.device
    H, W = model.H, model.W
    margin = model.tile_size // 2
    use_J = _has_jacobian(model)

    with torch.no_grad():
        mu = model.mu.detach()
        opacity = torch.sigmoid(model.opacity_logit).squeeze(-1)
        scales = torch.exp(model.log_scale)
        max_scale = scales.max(dim=1).values
        min_scale = scales.min(dim=1).values.clamp(min=1e-6)
        anisotropy = max_scale / min_scale
        finite_mask = (
            torch.isfinite(mu).all(dim=1) &
            torch.isfinite(scales).all(dim=1) &
            torch.isfinite(opacity)
        )

        residual_map, valid_map = cmax_loss_fn.compute_block_temporal_residual(
            events=events,
            flow=pred_flow,
            image_size=(H, W),
            block_size=residual_block_size,
            min_events_per_block=residual_min_events_per_block,
            use_polarity=False,
            no_grad=True,
        )

        coverage_map = torch.log1p(weight_map.squeeze(0).detach())
        coverage_block = block_reduce_map(coverage_map, residual_block_size, reduction='mean')

        curvature_map = compute_flow_curvature(pred_flow)
        curvature_map = F.avg_pool2d(
            torch.log1p(curvature_map).unsqueeze(0).unsqueeze(0),
            kernel_size=5,
            stride=1,
            padding=2,
        ).squeeze(0).squeeze(0)
        curvature_block = block_reduce_map(curvature_map, residual_block_size, reduction='mean')

        residual_badness_norm = 1.0 - robust_normalize(residual_map, valid_map)
        coverage_norm = robust_normalize(coverage_block, valid_map)
        curvature_norm = robust_normalize(curvature_block, valid_map)

        mask_nonfinite = ~finite_mask
        mask_geometry = (
            (max_scale < 0.5) |
            (max_scale > H / 6) |
            (anisotropy > max_anisotropy) |
            (mu[:, 0] < -margin) | (mu[:, 0] > H + margin) |
            (mu[:, 1] < -margin) | (mu[:, 1] > W + margin)
        )
        mask_prune = mask_nonfinite | mask_geometry | (opacity < prune_opacity_thresh)
        valid_parent = ~mask_prune

        # ══════ Clone：低局部方差 + 低覆盖 ══════
        n_clone = 0
        clone_dict = None
        clone_block_mask = None
        clone_score = None
        clone_from_parent = 0
        clone_random_init = 0
        clone_parent_radius = max(model.tile_size * 1.75, residual_block_size * 2.5)

        if 'clone' in phase and valid_map.any():
            res_vals = residual_map[valid_map]
            cov_vals = coverage_block[valid_map]
            res_thresh = torch.quantile(res_vals, clone_residual_quantile)
            cov_thresh = torch.quantile(cov_vals, clone_coverage_quantile)

            clone_block_mask = (
                valid_map &
                (residual_map <= res_thresh) &
                (coverage_block <= cov_thresh)
            )
            clone_score = residual_badness_norm * (1.0 - coverage_norm)
            candidate_flat = torch.where(clone_block_mask.view(-1))[0]

            if candidate_flat.numel() > 0:
                k = min(max_clone, candidate_flat.numel())
                top_local = torch.topk(clone_score.view(-1)[candidate_flat], k=k, largest=True).indices
                top_blocks = candidate_flat[top_local]
                block_centers = block_indices_to_centers(
                    top_blocks, clone_score.shape, residual_block_size, device
                )

                # mu_new = block_centers + torch.randn_like(block_centers) * clone_jitter
                mu_new = block_centers
                mu_new[:, 0].clamp_(0.0, H - 1.0)
                mu_new[:, 1].clamp_(0.0, W - 1.0)

                n_sel = mu_new.shape[0]
                clone_log_scale = torch.empty((n_sel, 2), device=device, dtype=model.log_scale.dtype)
                clone_rotation = torch.empty((n_sel, 1), device=device, dtype=model.rotation.dtype)
                clone_opacity = torch.empty((n_sel, 1), device=device, dtype=model.opacity_logit.dtype)
                clone_velocity = torch.empty((n_sel, 2), device=device, dtype=model.local_velocity.dtype)
                jacobian_dim = model.J_params.shape[1] if use_J else 0
                clone_jacobian = torch.empty(
                    (n_sel, jacobian_dim), device=device, dtype=model.J_params.dtype
                ) if use_J else None

                in_range = torch.zeros(n_sel, dtype=torch.bool, device=device)
                nearest_parent = None
                if valid_parent.any():
                    parent_idx_all = torch.where(valid_parent)[0]
                    parent_mu = mu[parent_idx_all]
                    dmat = torch.cdist(block_centers, parent_mu)
                    nearest_dist, nearest_local = dmat.min(dim=1)
                    nearest_parent = parent_idx_all[nearest_local]
                    in_range = nearest_dist < clone_parent_radius

                if in_range.any():
                    parent_sel = nearest_parent[in_range]
                    clone_log_scale[in_range] = model.log_scale.data[parent_sel].clone()
                    clone_rotation[in_range] = model.rotation.data[parent_sel].clone()
                    clone_opacity[in_range] = model.opacity_logit.data[parent_sel].clone()
                    clone_velocity[in_range] = model.local_velocity.data[parent_sel].clone()
                    if use_J:
                        clone_jacobian[in_range] = model.J_params.data[parent_sel].clone()
                    clone_from_parent = int(in_range.sum().item())

                random_mask = ~in_range
                if random_mask.any():
                    n_rand = int(random_mask.sum().item())
                    clone_log_scale[random_mask] = torch.stack([
                        torch.rand(n_rand, device=device, dtype=model.log_scale.dtype) * np.log(10.0),
                        torch.rand(n_rand, device=device, dtype=model.log_scale.dtype) * np.log(7.0),
                    ], dim=1)
                    clone_rotation[random_mask] = torch.rand(
                        n_rand, 1, device=device, dtype=model.rotation.dtype
                    ) * torch.pi
                    clone_opacity[random_mask] = torch.zeros(
                        n_rand, 1, device=device, dtype=model.opacity_logit.dtype
                    )
                    clone_velocity[random_mask] = torch.randn(
                        n_rand, 2, device=device, dtype=model.local_velocity.dtype
                    )
                    if use_J:
                        clone_jacobian[random_mask] = 0.05 * torch.randn(
                            n_rand, jacobian_dim, device=device, dtype=model.J_params.dtype
                        )
                    clone_random_init = n_rand

                clone_dict = {
                    'mu': mu_new,
                    'scale': clone_log_scale,
                    'rotation': clone_rotation,
                    'opacity': clone_opacity,
                    'velocity': clone_velocity,
                    'jacobian': clone_jacobian if use_J else None,
                }
                n_clone = n_sel

        # ══════ Split：低局部方差 + 高曲率 + 大高斯（基于支撑域聚合，不只看中心） ══════
        n_split = 0
        split_dict = None
        split_mask = None
        residual_support = None
        curvature_support = None
        scale_norm = None

        if 'split' in phase and valid_parent.any():
            _, _, residual_support = aggregate_block_map_over_gaussian_support(
                residual_map,
                mu,
                scales,
                model.rotation.detach(),
                residual_block_size,
                sample_steps=split_support_samples,
                support_extent=split_support_extent,
            )
            _, _, curvature_support = aggregate_block_map_over_gaussian_support(
                curvature_block,
                mu,
                scales,
                model.rotation.detach(),
                residual_block_size,
                sample_steps=split_support_samples,
                support_extent=split_support_extent,
            )

            residual_support_badness_norm = 1.0 - robust_normalize(residual_support, valid_parent)
            curvature_support_norm = robust_normalize(curvature_support, valid_parent)
            scale_norm = robust_normalize(max_scale, valid_parent)

            def _quantile(vals, q):
                q = float(max(0.0, min(1.0, q)))
                return torch.quantile(vals, q)

            valid_res = residual_support[valid_parent]
            valid_curv = curvature_support[valid_parent]
            valid_scale = max_scale[valid_parent]

            res_thresh = _quantile(valid_res, split_residual_quantile)
            curv_thresh = _quantile(valid_curv, split_curvature_quantile)
            scale_thresh = _quantile(valid_scale, split_scale_quantile)

            split_mask = (
                valid_parent &
                (residual_support <= res_thresh) &
                (curvature_support >= curv_thresh) &
                (max_scale >= scale_thresh)
            )

            split_score = residual_support_badness_norm * curvature_support_norm * scale_norm
            split_indices = torch.where(split_mask)[0]


            if split_indices.numel() > 0:
                k = min(max_split, split_indices.numel())
                top_local = torch.topk(split_score[split_indices], k=k, largest=True).indices
                split_indices = split_indices[top_local]
                n_split = split_indices.shape[0]

                mu_center = model.mu.data[split_indices].clone()
                r = model.rotation.data[split_indices].squeeze(-1)
                axis = torch.stack([torch.cos(r), torch.sin(r)], dim=-1)

                child_log_scale = model.log_scale.data[split_indices].clone() + float(np.log(split_scale_shrink))
                child_scale = torch.exp(child_log_scale)
                child_major = child_scale.max(dim=1, keepdim=True).values

                offset = axis * (split_offset_factor * child_major)

                parent_new_mu = mu_center + offset
                child_mu = mu_center - offset
                parent_new_mu[:, 0].clamp_(0.0, H - 1.0)
                parent_new_mu[:, 1].clamp_(0.0, W - 1.0)
                child_mu[:, 0].clamp_(0.0, H - 1.0)
                child_mu[:, 1].clamp_(0.0, W - 1.0)

                model.mu.data[split_indices] = parent_new_mu
                model.log_scale.data[split_indices] = child_log_scale.clone()
                model.opacity_logit.data[split_indices] = model.opacity_logit.data[split_indices]

                split_dict = {
                    'mu': child_mu,
                    'scale': child_log_scale.clone(),
                    'rotation': model.rotation.data[split_indices].clone(),
                    'opacity': model.opacity_logit.data[split_indices].clone(),
                    'velocity': model.local_velocity.data[split_indices].clone(),
                    'jacobian': model.J_params.data[split_indices].clone() if use_J else None,
                }

        if verbose:
            valid_blocks = int(valid_map.sum().item())
            res_mean = residual_map[valid_map].mean().item() if valid_map.any() else 0.0
            cov_mean = coverage_block[valid_map].mean().item() if valid_map.any() else 0.0
            curv_mean = curvature_block[valid_map].mean().item() if valid_map.any() else 0.0
            print(
                f"  [DensityCtrl] valid_blocks={valid_blocks}  "
                f"localvar_mean={res_mean:.4f}  cov_mean={cov_mean:.4f}  curv_mean={curv_mean:.4f}"
            )
            if 'clone' in phase:
                clone_cands = int(clone_block_mask.sum().item()) if clone_block_mask is not None else 0
                print(
                    f"  [Clone] block_candidates={clone_cands}  selected={n_clone}  "
                    f"copied={clone_from_parent}  random={clone_random_init}  radius={clone_parent_radius:.1f}"
                )
            if 'split' in phase:
                split_cands = int(split_mask.sum().item()) if split_mask is not None else 0
                if residual_support is not None and curvature_support is not None:
                    print(
                        f"gaussian_candidates={split_cands}  selected={n_split}  "
                        f"res_support_mean={residual_support[valid_parent].mean().item():.4f}  "
                        f"curv_support_mean={curvature_support[valid_parent].mean().item():.4f}"
                    )
                else:
                    print(f"gaussian_candidates={split_cands}  selected={n_split}")
            if 'prune' in phase:
                print(
                    f"  [Prune] opacity={int((opacity < prune_opacity_thresh).sum().item())}  "
                    f"geometry={int(mask_geometry.sum().item())}  nonfinite={int(mask_nonfinite.sum().item())}  "
                    f"total={int(mask_prune.sum().item())}"
                )

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
        current_N = model.num_gaussians
        prune_full = torch.zeros(current_N, dtype=torch.bool, device=device)
        prune_full[:mu.shape[0]] = mask_prune
        if prune_full.any():
            n_pruned = int(prune_full.sum().item())
            prune_gaussians(model, optimizer, prune_full)
            has_changed = True

    if verbose:
        print(f"  [{phase}] Clone:{n_clone}  Split:{n_split}  Prune:{n_pruned}  -> N={model.num_gaussians}")

    return has_changed


# ─────────────────────────────────────────────────────────────────────────────
#  训练历史可视化
# ─────────────────────────────────────────────────────────────────────────────
