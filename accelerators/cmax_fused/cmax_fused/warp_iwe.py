import torch
from . import cmax_fused_cuda_ext


class _FusedWarpIWEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, events, flow, H, W, t_ref, sigma, kernel_size):
        """
        events: [N, 4] float32 cuda, 格式 [h, w, t, p]
        flow  : [2, H, W] float32 cuda
        返回  : [2, H, W]，channel 0 = neg, channel 1 = pos
        """
        events = events.contiguous()
        flow = flow.contiguous()

        iwe = cmax_fused_cuda_ext.forward(
            events, flow, int(H), int(W), float(t_ref), float(sigma), int(kernel_size)
        )

        ctx.save_for_backward(events, flow)
        ctx.H = int(H)
        ctx.W = int(W)
        ctx.t_ref = float(t_ref)
        ctx.sigma = float(sigma)
        ctx.kernel_size = int(kernel_size)
        return iwe

    @staticmethod
    def backward(ctx, grad_iwe):
        events, flow = ctx.saved_tensors
        grad_iwe = grad_iwe.contiguous()

        grad_flow = cmax_fused_cuda_ext.backward(
            grad_iwe, events, flow,
            ctx.H, ctx.W, ctx.t_ref, ctx.sigma, ctx.kernel_size
        )

        return None, grad_flow, None, None, None, None, None


def fused_warp_iwe_cuda(events, flow, image_size, t_ref, sigma, kernel_size):
    H, W = image_size
    return _FusedWarpIWEFunction.apply(
        events, flow, H, W, t_ref, sigma, kernel_size
    )