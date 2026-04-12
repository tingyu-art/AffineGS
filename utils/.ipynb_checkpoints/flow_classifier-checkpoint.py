import torch
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────
# centroid
# ─────────────────────────────────────────────
def _centroid(density: torch.Tensor):
    H, W  = density.shape
    dev   = density.device

    h_idx = torch.arange(H, device=dev).float()
    w_idx = torch.arange(W, device=dev).float()

    total = density.sum() + 1e-8
    ch = (density * h_idx[:, None]).sum() / total
    cw = (density * w_idx[None, :]).sum() / total

    return ch, cw


# ─────────────────────────────────────────────
# feature 1
# ─────────────────────────────────────────────
def _feat_density_uniformity(density: torch.Tensor) -> float:
    mean = density.mean()
    if mean < 1e-6:
        return 0.5
    cv = (density.std() / (mean + 1e-8)).item()
    return float(np.exp(-3.0 * cv))


# ─────────────────────────────────────────────
# feature 2
# ─────────────────────────────────────────────
def _feat_radial_peak(density: torch.Tensor, n_bins: int = 20) -> float:
    H, W   = density.shape
    dev    = density.device
    ch, cw = _centroid(density)

    h_idx  = torch.arange(H, device=dev).float()
    w_idx  = torch.arange(W, device=dev).float()
    hh, ww = torch.meshgrid(h_idx, w_idx, indexing='ij')

    r      = torch.sqrt((hh - ch) ** 2 + (ww - cw) ** 2).reshape(-1)
    d_flat = density.reshape(-1)

    r_max = min(H, W) * 0.45
    valid = r < r_max

    r_v = r[valid]
    d_v = d_flat[valid]

    edges   = torch.linspace(0, r_max, n_bins + 1, device=dev)
    profile = []

    for i in range(n_bins):
        m = (r_v >= edges[i]) & (r_v < edges[i + 1])
        profile.append(d_v[m].mean().item() if m.sum() > 10 else 0.0)

    p = np.array(profile)
    p /= (p.max() + 1e-8)

    peak_idx    = int(np.argmax(p))
    is_interior = 2 <= peak_idx <= n_bins - 3

    edge_mean = (p[:2].mean() + p[-2:].mean()) / 2
    score     = float(p[peak_idx] - edge_mean) * float(is_interior)

    return float(np.clip(score * 2.0, 0.0, 1.0))


# ─────────────────────────────────────────────
# feature 3
# ─────────────────────────────────────────────
def _feat_ring_structure(density: torch.Tensor) -> float:
    H, W   = density.shape
    dev    = density.device
    ch, cw = _centroid(density)

    h_idx  = torch.arange(H, device=dev).float()
    w_idx  = torch.arange(W, device=dev).float()
    hh, ww = torch.meshgrid(h_idx, w_idx, indexing='ij')

    r     = torch.sqrt((hh - ch) ** 2 + (ww - cw) ** 2)
    theta = torch.atan2(hh - ch, ww - cw)

    r_max   = min(H, W) * 0.45
    r_bands = [(r_max * 0.05, r_max * 0.35),
               (r_max * 0.15, r_max * 0.55),
               (r_max * 0.30, r_max * 0.70)]

    n_ang = 16
    edges = torch.linspace(-float(torch.pi), float(torch.pi), n_ang + 1, device=dev)

    min_cv = 1.0

    for r_lo, r_hi in r_bands:
        ring_mask = (r >= r_lo) & (r <= r_hi)

        sector_means = []
        for i in range(n_ang):
            m = ring_mask & (theta >= edges[i]) & (theta < edges[i + 1])
            if m.sum() > 10:
                sector_means.append(density[m].mean().item())

        if len(sector_means) >= 10:
            arr = np.array(sector_means)
            cv  = arr.std() / (arr.mean() + 1e-8)
            min_cv = min(min_cv, cv)

    return float(np.clip(1.0 - min_cv * 3.0, 0.0, 1.0))


# ─────────────────────────────────────────────
# feature 4
# ─────────────────────────────────────────────
def _feat_axial_asymmetry(density: torch.Tensor) -> float:
    ph    = torch.fft.rfft(density.mean(dim=1)).abs() ** 2
    pw    = torch.fft.rfft(density.mean(dim=0)).abs() ** 2

    ph[0] = 0
    pw[0] = 0

    total = ph.sum() + pw.sum() + 1e-8
    return float((torch.abs(ph.sum() - pw.sum()) / total).item())


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def is_turbulence(
    density: torch.Tensor,
    smooth_k: int = 0,
) -> bool:
    """
    输入任意尺寸 (H, W) 的事件图，判断是否为湍流
    """

    # 自动读取尺寸（核心改动）
    assert density.ndim == 2, "density must be [H, W]"
    H, W = density.shape

    # smoothing（保持原逻辑）
    if smooth_k > 1:
        density = F.avg_pool2d(
            density[None, None],
            smooth_k,
            stride=1,
            padding=smooth_k // 2
        ).squeeze()

    # features（完全不变）
    unif = _feat_density_uniformity(density)
    rpk  = _feat_radial_peak(density)
    ring = _feat_ring_structure(density)
    asym = _feat_axial_asymmetry(density)

    # 原 decision tree（完全不变）
    if unif > 0.40:
        return False

    if asym > 0.70:
        return False

    if ring < 0.55 and rpk < 0.80:
        return True

    if rpk < 0.20:
        return False

    return False