#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be int64")
#define CHECK_INPUT_FLOAT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_FLOAT(x)
#define CHECK_INPUT_LONG(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_LONG(x)

std::vector<torch::Tensor> gaussian_render_forward_cuda(
    torch::Tensor mu,
    torch::Tensor sigma_inv,
    torch::Tensor opacity,
    torch::Tensor vel,
    torch::Tensor J,
    torch::Tensor cell_ptr,
    torch::Tensor cell_indices,
    int64_t H,
    int64_t W,
    int64_t tile_size,
    int64_t grid_w);

std::vector<torch::Tensor> gaussian_render_backward_cuda(
    torch::Tensor grad_num_h,
    torch::Tensor grad_num_w,
    torch::Tensor grad_den,
    torch::Tensor mu,
    torch::Tensor sigma_inv,
    torch::Tensor opacity,
    torch::Tensor vel,
    torch::Tensor J,
    torch::Tensor cell_ptr,
    torch::Tensor cell_indices,
    int64_t H,
    int64_t W,
    int64_t tile_size,
    int64_t grid_w);

std::vector<torch::Tensor> gaussian_render_forward(
    torch::Tensor mu,
    torch::Tensor sigma_inv,
    torch::Tensor opacity,
    torch::Tensor vel,
    torch::Tensor J,
    torch::Tensor cell_ptr,
    torch::Tensor cell_indices,
    int64_t H,
    int64_t W,
    int64_t tile_size,
    int64_t grid_w) {
    CHECK_INPUT_FLOAT(mu);
    CHECK_INPUT_FLOAT(sigma_inv);
    CHECK_INPUT_FLOAT(opacity);
    CHECK_INPUT_FLOAT(vel);
    CHECK_INPUT_FLOAT(J);
    CHECK_INPUT_LONG(cell_ptr);
    CHECK_INPUT_LONG(cell_indices);

    TORCH_CHECK(mu.dim() == 2 && mu.size(1) == 2, "mu must have shape [N, 2]");
    TORCH_CHECK(sigma_inv.dim() == 3 && sigma_inv.size(1) == 2 && sigma_inv.size(2) == 2,
                "sigma_inv must have shape [N, 2, 2]");
    TORCH_CHECK(opacity.dim() == 2 && opacity.size(1) == 1, "opacity must have shape [N, 1]");
    TORCH_CHECK(vel.dim() == 2 && vel.size(1) == 2, "vel must have shape [N, 2]");
    TORCH_CHECK(J.dim() == 3 && J.size(1) == 2 && J.size(2) == 2, "J must have shape [N, 2, 2]");

    return gaussian_render_forward_cuda(mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices,
                                        H, W, tile_size, grid_w);
}

std::vector<torch::Tensor> gaussian_render_backward(
    torch::Tensor grad_num_h,
    torch::Tensor grad_num_w,
    torch::Tensor grad_den,
    torch::Tensor mu,
    torch::Tensor sigma_inv,
    torch::Tensor opacity,
    torch::Tensor vel,
    torch::Tensor J,
    torch::Tensor cell_ptr,
    torch::Tensor cell_indices,
    int64_t H,
    int64_t W,
    int64_t tile_size,
    int64_t grid_w) {
    CHECK_INPUT_FLOAT(grad_num_h);
    CHECK_INPUT_FLOAT(grad_num_w);
    CHECK_INPUT_FLOAT(grad_den);
    CHECK_INPUT_FLOAT(mu);
    CHECK_INPUT_FLOAT(sigma_inv);
    CHECK_INPUT_FLOAT(opacity);
    CHECK_INPUT_FLOAT(vel);
    CHECK_INPUT_FLOAT(J);
    CHECK_INPUT_LONG(cell_ptr);
    CHECK_INPUT_LONG(cell_indices);

    return gaussian_render_backward_cuda(grad_num_h, grad_num_w, grad_den,
                                         mu, sigma_inv, opacity, vel, J,
                                         cell_ptr, cell_indices,
                                         H, W, tile_size, grid_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gaussian_render_forward, "Gaussian tile render forward (CUDA)");
    m.def("backward", &gaussian_render_backward, "Gaussian tile render backward (CUDA)");
}
