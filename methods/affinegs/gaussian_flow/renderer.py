"""Rendering backend adapters for Gaussian flow fields."""

import torch
import warnings

try:
    from ..accelerators.gaussian_renderer import gaussian_render_cuda, gaussian_render_derivatives_cuda

    HAS_CUDA_RENDERER = True
    _CUDA_IMPORT_ERROR = None
except Exception as exc:
    gaussian_render_cuda = None
    gaussian_render_derivatives_cuda = None
    HAS_CUDA_RENDERER = False
    _CUDA_IMPORT_ERROR = exc


def initialize_cuda_renderer(device, on_error="error"):
    """Resolve the backend before allocating training state; never silently switch.

    Import success alone is not readiness: the JIT module must load and expose all
    required entry points. Runtime/kernel errors after initialization propagate.
    """
    if on_error not in ("error", "fallback"):
        raise ValueError("cuda_renderer_policy must be 'error' or 'fallback'")
    device = torch.device(device)
    if device.type != 'cuda':
        return "torch"
    try:
        if not HAS_CUDA_RENDERER:
            raise RuntimeError("CUDA renderer import failed") from _CUDA_IMPORT_ERROR
        # Resolve the module dynamically so notebook dependency reloads also
        # refresh the cached extension loader used here.
        from ..accelerators.gaussian_renderer import renderer as cuda_renderer
        with torch.cuda.device(device):
            extension = cuda_renderer._load_ext()
        if not all(callable(getattr(extension, name, None))
                   for name in ("forward", "backward", "derivatives")):
            raise RuntimeError("CUDA renderer extension has missing entry points; rebuild it")
    except Exception as exc:
        message = (
            "Gaussian CUDA renderer initialization failed before training. "
            "Check the CUDA toolkit/compiler and rebuild the extension. "
            "To explicitly permit the PyTorch backend, set cuda_renderer_policy='fallback'. "
            f"Original error: {exc}"
        )
        if on_error == "error":
            raise RuntimeError(message) from exc
        warnings.warn(message + " Using the PyTorch renderer (render_backend='torch').",
                      RuntimeWarning, stacklevel=2)
        return "torch"
    return "cuda"


def render_gaussians_cuda(
    mu: torch.Tensor,
    sigma_inv: torch.Tensor,
    opacity: torch.Tensor,
    velocity: torch.Tensor,
    jacobian: torch.Tensor,
    cell_ptr: torch.Tensor,
    cell_indices: torch.Tensor,
    height: int,
    width: int,
    tile_size: int,
    grid_width: int,
):
    """Render Gaussian velocity accumulators with the optional CUDA extension."""
    if not HAS_CUDA_RENDERER:
        raise RuntimeError("Gaussian flow CUDA renderer is not available")

    return gaussian_render_cuda(
        mu,
        sigma_inv,
        opacity,
        velocity,
        jacobian,
        cell_ptr,
        cell_indices,
        height,
        width,
        tile_size,
        grid_width,
    )
