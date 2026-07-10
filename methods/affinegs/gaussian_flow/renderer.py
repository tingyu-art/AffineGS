"""Rendering backend adapters for Gaussian flow fields."""

import torch

try:
    from ..accelerators.gaussian_renderer import gaussian_render_cuda

    HAS_CUDA_RENDERER = True
except Exception:
    gaussian_render_cuda = None
    HAS_CUDA_RENDERER = False


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
