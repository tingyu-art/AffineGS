#include <torch/extension.h>

torch::Tensor fused_warp_iwe_forward_cuda(
    torch::Tensor events,
    torch::Tensor flow,
    int64_t H,
    int64_t W,
    double t_ref,
    double sigma,
    int64_t kernel_size);

torch::Tensor fused_warp_iwe_backward_cuda(
    torch::Tensor grad_iwe,
    torch::Tensor events,
    torch::Tensor flow,
    int64_t H,
    int64_t W,
    double t_ref,
    double sigma,
    int64_t kernel_size);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT32(x)

torch::Tensor fused_warp_iwe_forward(
    torch::Tensor events,
    torch::Tensor flow,
    int64_t H,
    int64_t W,
    double t_ref,
    double sigma,
    int64_t kernel_size) {

    CHECK_INPUT(events);
    CHECK_INPUT(flow);

    TORCH_CHECK(events.dim() == 2 && events.size(1) == 4, "events must be [N,4]");
    TORCH_CHECK(flow.dim() == 3 && flow.size(0) == 2, "flow must be [2,H,W]");
    TORCH_CHECK(flow.size(1) == H && flow.size(2) == W, "flow shape mismatch with H,W");
    TORCH_CHECK(kernel_size >= 1 && (kernel_size % 2 == 1), "kernel_size must be odd");
    TORCH_CHECK(sigma > 0.0, "sigma must be > 0");

    return fused_warp_iwe_forward_cuda(events, flow, H, W, t_ref, sigma, kernel_size);
}

torch::Tensor fused_warp_iwe_backward(
    torch::Tensor grad_iwe,
    torch::Tensor events,
    torch::Tensor flow,
    int64_t H,
    int64_t W,
    double t_ref,
    double sigma,
    int64_t kernel_size) {

    CHECK_INPUT(grad_iwe);
    CHECK_INPUT(events);
    CHECK_INPUT(flow);

    TORCH_CHECK(grad_iwe.dim() == 3 && grad_iwe.size(0) == 2, "grad_iwe must be [2,H,W]");
    TORCH_CHECK(events.dim() == 2 && events.size(1) == 4, "events must be [N,4]");
    TORCH_CHECK(flow.dim() == 3 && flow.size(0) == 2, "flow must be [2,H,W]");
    TORCH_CHECK(flow.size(1) == H && flow.size(2) == W, "flow shape mismatch with H,W");
    TORCH_CHECK(kernel_size >= 1 && (kernel_size % 2 == 1), "kernel_size must be odd");
    TORCH_CHECK(sigma > 0.0, "sigma must be > 0");

    return fused_warp_iwe_backward_cuda(
        grad_iwe, events, flow, H, W, t_ref, sigma, kernel_size
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_warp_iwe_forward, "fused warp + iwe forward (CUDA)");
    m.def("backward", &fused_warp_iwe_backward, "fused warp + iwe backward (CUDA)");
}