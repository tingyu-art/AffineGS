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

torch::Tensor gaussian_render_derivatives_cuda(
    torch::Tensor mu, torch::Tensor sigma_inv, torch::Tensor opacity,
    torch::Tensor vel, torch::Tensor J, torch::Tensor cell_ptr, torch::Tensor cell_indices,
    int64_t H, int64_t W, int64_t tile_size, int64_t grid_w);

namespace {
void check_renderer_inputs(
    const torch::Tensor& mu, const torch::Tensor& sigma_inv, const torch::Tensor& opacity,
    const torch::Tensor& vel, const torch::Tensor& J,
    const torch::Tensor& cell_ptr, const torch::Tensor& cell_indices,
    int64_t H, int64_t W, int64_t tile_size, int64_t grid_w) {
    CHECK_INPUT_FLOAT(mu);
    CHECK_INPUT_FLOAT(sigma_inv);
    CHECK_INPUT_FLOAT(opacity);
    CHECK_INPUT_FLOAT(vel);
    CHECK_INPUT_FLOAT(J);
    CHECK_INPUT_LONG(cell_ptr);
    CHECK_INPUT_LONG(cell_indices);
    TORCH_CHECK(mu.dim() == 2 && mu.size(1) == 2, "mu must have shape [N, 2]");
    const auto n = mu.size(0);
    TORCH_CHECK(sigma_inv.sizes() == torch::IntArrayRef({n, 2, 2}), "sigma_inv must have shape [N, 2, 2]");
    TORCH_CHECK(opacity.sizes() == torch::IntArrayRef({n, 1}), "opacity must have shape [N, 1]");
    TORCH_CHECK(vel.sizes() == torch::IntArrayRef({n, 2}), "vel must have shape [N, 2]");
    TORCH_CHECK(J.sizes() == torch::IntArrayRef({n, 2, 2}), "J must have shape [N, 2, 2]");
    TORCH_CHECK(H > 0 && W > 0 && tile_size > 0, "image and tile dimensions must be positive");
    TORCH_CHECK(grid_w == (W + tile_size - 1) / tile_size, "grid_w does not match W and tile_size");
    TORCH_CHECK(cell_ptr.dim() == 1 && cell_ptr.numel() == ((H + tile_size - 1) / tile_size) * grid_w + 1,
                "cell_ptr does not match the tile grid");
    TORCH_CHECK(cell_indices.dim() == 1, "cell_indices must be one-dimensional");
    for (const auto& input : {sigma_inv, opacity, vel, J, cell_ptr, cell_indices}) {
        TORCH_CHECK(input.device() == mu.device(), "renderer inputs must share a CUDA device");
    }
}
} // namespace

torch::Tensor gaussian_render_derivatives(
    torch::Tensor mu, torch::Tensor sigma_inv, torch::Tensor opacity,
    torch::Tensor vel, torch::Tensor J, torch::Tensor cell_ptr, torch::Tensor cell_indices,
    int64_t H, int64_t W, int64_t tile_size, int64_t grid_w) {
    check_renderer_inputs(mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices,
                          H, W, tile_size, grid_w);

    return gaussian_render_derivatives_cuda(mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices,
                                            H, W, tile_size, grid_w);
}

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
    check_renderer_inputs(mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices,
                          H, W, tile_size, grid_w);

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
    check_renderer_inputs(mu, sigma_inv, opacity, vel, J, cell_ptr, cell_indices,
                          H, W, tile_size, grid_w);
    for (const auto& grad : {grad_num_h, grad_num_w, grad_den}) {
        CHECK_INPUT_FLOAT(grad);
        TORCH_CHECK(grad.device() == mu.device(), "output gradients must share the input CUDA device");
        TORCH_CHECK(grad.sizes() == torch::IntArrayRef({H, W}), "output gradients must have shape [H, W]");
    }

    return gaussian_render_backward_cuda(grad_num_h, grad_num_w, grad_den,
                                         mu, sigma_inv, opacity, vel, J,
                                         cell_ptr, cell_indices,
                                         H, W, tile_size, grid_w);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gaussian_render_forward, "Gaussian tile render forward (CUDA)");
    m.def("backward", &gaussian_render_backward, "Gaussian tile render backward (CUDA)");
    m.def("derivatives", &gaussian_render_derivatives, "Gaussian tile render with analytic derivatives (CUDA, inference only)");
}
