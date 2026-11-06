"""Input contracts shared by public estimation entry points; no data coercion."""
import math
import numbers
import numpy as np
import torch


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_events(events, H=None, W=None, require_span=True):
    """Validate [row, column, milliseconds, polarity], without reordering/casting."""
    if not isinstance(events, (torch.Tensor, np.ndarray)):
        raise TypeError("events must be a tensor or numpy array with shape [N, 4]")
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must have shape [N, 4]: row, column, time_ms, polarity")
    if len(events) == 0:
        raise ValueError("events must contain at least one event")
    finite = torch.isfinite(events).all() if torch.is_tensor(events) else np.isfinite(events).all()
    if not bool(finite):
        raise ValueError("events must contain only finite values")
    for axis, extent, label in [(0, H, 'row'), (1, W, 'column')]:
        if extent is None:
            continue
        positive_int(extent, 'H' if axis == 0 else 'W')
        if bool(((events[:, axis] < 0) | (events[:, axis] >= extent)).any()):
            raise ValueError(f"event {label} coordinates must lie in [0, {extent})")
    if require_span and not bool(events[:, 2].max() > events[:, 2].min()):
        raise ValueError("events require at least two distinct timestamps")


def validate_gs_config(cfg):
    for name in ['H', 'W', 'init_n', 'tile_size', 'iterations', 'density_interval']:
        positive_int(getattr(cfg, name), name)
    if not isinstance(cfg.phase, (tuple, list)) or len(cfg.phase) != 3:
        raise ValueError("phase must contain exactly three nonnegative integer thresholds")
    if any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0 for v in cfg.phase):
        raise ValueError("phase thresholds must be nonnegative integers")
    # Existing ablations intentionally place phase[0] beyond all iterations,
    # e.g. [500, 200, 250]. Density control is then inactive; do not reorder it.
    if cfg.enable_density_control and cfg.phase[0] < cfg.iterations:
        if not cfg.phase[0] <= cfg.phase[1] <= cfg.phase[2]:
            raise ValueError("active density-control phase thresholds must be ordered")
    for name in ['lr_mu', 'cmax_sigma', 'init_scale_h', 'init_scale_w']:
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if cfg.lr_j is not None and (not math.isfinite(float(cfg.lr_j)) or cfg.lr_j <= 0):
        raise ValueError("lr_j must be positive and finite, or None for auto")
    for name in ['cmax_kernel_size', 'density_smooth_k']:
        value = getattr(cfg, name)
        positive_int(value, name)
        if value % 2 != 1:
            raise ValueError(f"{name} must be odd")
    for name in ['div_top_ratio', 'prune_opacity_thresh']:
        value = float(getattr(cfg, name))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if not math.isfinite(float(cfg.w_div)) or cfg.w_div < 0:
        raise ValueError("w_div must be finite and nonnegative")
    if not math.isfinite(float(cfg.log_scale_min)) or not 0 <= cfg.log_scale_min <= math.log(cfg.H / 6):
        raise ValueError("log_scale_min must lie between log(1px) and log(H/6)")


def validate_flow(flow, H, W, name='flow'):
    if not torch.is_tensor(flow) or tuple(flow.shape) != (2, H, W):
        raise ValueError(f"{name} must be a tensor with shape [2, {H}, {W}]")
    if not bool(torch.isfinite(flow).all()):
        raise ValueError(f"{name} must contain only finite values")
