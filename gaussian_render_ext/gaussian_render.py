from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load

_THIS_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def _load_ext():
    return load(
        name="gaussian_render_ext",
        sources=[
            str(_THIS_DIR / "gaussian_render.cpp"),
            str(_THIS_DIR / "gaussian_render_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        # no --use_fast_math: keep float32 math behavior strict
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


class GaussianRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices, H, W, tile_size, grid_w):
        ext = _load_ext()

        mu = mu.contiguous().float()
        sigma_inv = sigma_inv.contiguous().float()
        opacity = opacity.contiguous().float()
        vel = vel.contiguous().float()
        J = J.contiguous().float()
        cell_ptr = cell_ptr.contiguous().long()
        cell_indices = cell_indices.contiguous().long()

        num_h, num_w, den = ext.forward(
            mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices,
            int(H), int(W), int(tile_size), int(grid_w)
        )

        ctx.save_for_backward(mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices)
        ctx.H = int(H)
        ctx.W = int(W)
        ctx.tile_size = int(tile_size)
        ctx.grid_w = int(grid_w)
        return num_h, num_w, den

    @staticmethod
    def backward(ctx, grad_num_h, grad_num_w, grad_den):
        ext = _load_ext()
        mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices = ctx.saved_tensors

        grad_num_h = grad_num_h.contiguous().float()
        grad_num_w = grad_num_w.contiguous().float()
        grad_den = grad_den.contiguous().float()

        grad_mu, grad_sigma_inv, grad_opacity, grad_vel, grad_J = ext.backward(
            grad_num_h, grad_num_w, grad_den,
            mu, sigma_inv, opacity, vel, J,
            cell_ptr, cell_indices,
            ctx.H, ctx.W, ctx.tile_size, ctx.grid_w,
        )

        return grad_mu, grad_sigma_inv, grad_opacity, grad_vel, grad_J, None, None, None, None, None, None


def gaussian_render_cuda(mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices, H, W, tile_size, grid_w):
    return GaussianRenderFunction.apply(mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices,
                                        H, W, tile_size, grid_w)
