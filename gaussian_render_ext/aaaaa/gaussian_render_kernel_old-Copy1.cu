#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

constexpr float kGateCenter = 14.0f;
constexpr float kGateSlope = -2.0f;

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gaussian_render_forward_kernel(
    const float* __restrict__ mu,           // [N, 2]
    const float* __restrict__ sigma_inv,    // [N, 2, 2]
    const float* __restrict__ opacity,      // [N, 1]
    const float* __restrict__ vel,          // [N, 2]
    const float* __restrict__ J,            // [N, 2, 2]
    const int64_t* __restrict__ cell_ptr,   // [num_tiles + 1]
    const int64_t* __restrict__ cell_idx,   // [pairs]
    int H,
    int W,
    int tile_size,
    int grid_w,
    float* __restrict__ out_num_h,
    float* __restrict__ out_num_w,
    float* __restrict__ out_den) {

    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = H * W;
    if (pix >= HW) return;

    const int ph = pix / W;
    const int pw = pix - ph * W;
    const int tile_r = ph / tile_size;
    const int tile_c = pw / tile_size;
    const int tile_id = tile_r * grid_w + tile_c;

    const int64_t start = cell_ptr[tile_id];
    const int64_t end   = cell_ptr[tile_id + 1];

    float acc_num_h = 0.0f;
    float acc_num_w = 0.0f;
    float acc_den   = 0.0f;

    for (int64_t p = start; p < end; ++p) {
        const int64_t g = cell_idx[p];

        const float mu_h = mu[g * 2 + 0];
        const float mu_w = mu[g * 2 + 1];
        const float dh = static_cast<float>(ph) - mu_h;
        const float dw = static_cast<float>(pw) - mu_w;

        const float Si_hh = sigma_inv[g * 4 + 0];
        const float Si_hw = sigma_inv[g * 4 + 1];
        const float Si_ww = sigma_inv[g * 4 + 3];

        const float term = dh * dh * Si_hh + 2.0f * dh * dw * Si_hw + dw * dw * Si_ww;
        const float gate = sigmoidf_fast(kGateSlope * (term - kGateCenter));
        const float base = expf(-0.5f * term) * gate;
        const float w = base * opacity[g];

        const float v_h = vel[g * 2 + 0] + J[g * 4 + 0] * dh + J[g * 4 + 1] * dw;
        const float v_w = vel[g * 2 + 1] + J[g * 4 + 2] * dh + J[g * 4 + 3] * dw;

        acc_num_h += w * v_h;
        acc_num_w += w * v_w;
        acc_den   += w;
    }

    out_num_h[pix] = acc_num_h;
    out_num_w[pix] = acc_num_w;
    out_den[pix]   = acc_den;
}

__global__ void gaussian_render_backward_kernel(
    const float* __restrict__ grad_num_h,
    const float* __restrict__ grad_num_w,
    const float* __restrict__ grad_den,
    const float* __restrict__ mu,
    const float* __restrict__ sigma_inv,
    const float* __restrict__ opacity,
    const float* __restrict__ vel,
    const float* __restrict__ J,
    const int64_t* __restrict__ cell_ptr,
    const int64_t* __restrict__ cell_idx,
    int H,
    int W,
    int tile_size,
    int grid_w,
    float* __restrict__ grad_mu,
    float* __restrict__ grad_sigma_inv,
    float* __restrict__ grad_opacity,
    float* __restrict__ grad_vel,
    float* __restrict__ grad_J) {

    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    const int HW = H * W;
    if (pix >= HW) return;

    const float gNh = grad_num_h[pix];
    const float gNw = grad_num_w[pix];
    const float gD  = grad_den[pix];

    if (gNh == 0.0f && gNw == 0.0f && gD == 0.0f) return;

    const int ph = pix / W;
    const int pw = pix - ph * W;
    const int tile_r = ph / tile_size;
    const int tile_c = pw / tile_size;
    const int tile_id = tile_r * grid_w + tile_c;

    const int64_t start = cell_ptr[tile_id];
    const int64_t end   = cell_ptr[tile_id + 1];

    for (int64_t p = start; p < end; ++p) {
        const int64_t g = cell_idx[p];

        const float mu_h = mu[g * 2 + 0];
        const float mu_w = mu[g * 2 + 1];
        const float dh = static_cast<float>(ph) - mu_h;
        const float dw = static_cast<float>(pw) - mu_w;

        const float Si_hh = sigma_inv[g * 4 + 0];
        const float Si_hw = sigma_inv[g * 4 + 1];
        const float Si_ww = sigma_inv[g * 4 + 3];

        const float term = dh * dh * Si_hh + 2.0f * dh * dw * Si_hw + dw * dw * Si_ww;
        const float gate = sigmoidf_fast(kGateSlope * (term - kGateCenter));
        const float base = expf(-0.5f * term) * gate;
        const float op   = opacity[g];
        const float w    = base * op;

        const float J00 = J[g * 4 + 0];
        const float J01 = J[g * 4 + 1];
        const float J10 = J[g * 4 + 2];
        const float J11 = J[g * 4 + 3];

        const float v_h = vel[g * 2 + 0] + J00 * dh + J01 * dw;
        const float v_w = vel[g * 2 + 1] + J10 * dh + J11 * dw;

        const float dL_dw = gNh * v_h + gNw * v_w + gD;
        const float dL_dvh = gNh * w;
        const float dL_dvw = gNw * w;

        const float dbase_dterm = base * (2.0f * gate - 2.5f);
        const float dwgt_dterm = op * dbase_dterm;

        const float dterm_ddh = 2.0f * (Si_hh * dh + Si_hw * dw);
        const float dterm_ddw = 2.0f * (Si_hw * dh + Si_ww * dw);

        const float dL_ddh = dL_dw * dwgt_dterm * dterm_ddh + dL_dvh * J00 + dL_dvw * J10;
        const float dL_ddw = dL_dw * dwgt_dterm * dterm_ddw + dL_dvh * J01 + dL_dvw * J11;

        atomicAdd(&grad_mu[g * 2 + 0], -dL_ddh);
        atomicAdd(&grad_mu[g * 2 + 1], -dL_ddw);

        atomicAdd(&grad_sigma_inv[g * 4 + 0], dL_dw * dwgt_dterm * (dh * dh));
        atomicAdd(&grad_sigma_inv[g * 4 + 1], dL_dw * dwgt_dterm * (2.0f * dh * dw));
        // grad_sigma_inv[g * 4 + 2] keeps zero to match the Python path.
        atomicAdd(&grad_sigma_inv[g * 4 + 3], dL_dw * dwgt_dterm * (dw * dw));

        atomicAdd(&grad_opacity[g], dL_dw * base);

        atomicAdd(&grad_vel[g * 2 + 0], dL_dvh);
        atomicAdd(&grad_vel[g * 2 + 1], dL_dvw);

        atomicAdd(&grad_J[g * 4 + 0], dL_dvh * dh);
        atomicAdd(&grad_J[g * 4 + 1], dL_dvh * dw);
        atomicAdd(&grad_J[g * 4 + 2], dL_dvw * dh);
        atomicAdd(&grad_J[g * 4 + 3], dL_dvw * dw);
    }
}

} // namespace

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
    int64_t grid_w) {

    auto opts = mu.options();
    auto out_num_h = torch::zeros({H, W}, opts);
    auto out_num_w = torch::zeros({H, W}, opts);
    auto out_den   = torch::zeros({H, W}, opts);

    const int threads = 256;
    const int HW = static_cast<int>(H * W);
    const int blocks = (HW + threads - 1) / threads;

    gaussian_render_forward_kernel<<<blocks, threads>>>(
        mu.data_ptr<float>(),
        sigma_inv.data_ptr<float>(),
        opacity.data_ptr<float>(),
        vel.data_ptr<float>(),
        J.data_ptr<float>(),
        cell_ptr.data_ptr<int64_t>(),
        cell_indices.data_ptr<int64_t>(),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(tile_size),
        static_cast<int>(grid_w),
        out_num_h.data_ptr<float>(),
        out_num_w.data_ptr<float>(),
        out_den.data_ptr<float>());

    return {out_num_h, out_num_w, out_den};
}

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
    int64_t grid_w) {

    auto grad_mu        = torch::zeros_like(mu);
    auto grad_sigma_inv = torch::zeros_like(sigma_inv);
    auto grad_opacity   = torch::zeros_like(opacity);
    auto grad_vel       = torch::zeros_like(vel);
    auto grad_J         = torch::zeros_like(J);

    const int threads = 256;
    const int HW = static_cast<int>(H * W);
    const int blocks = (HW + threads - 1) / threads;

    gaussian_render_backward_kernel<<<blocks, threads>>>(
        grad_num_h.data_ptr<float>(),
        grad_num_w.data_ptr<float>(),
        grad_den.data_ptr<float>(),
        mu.data_ptr<float>(),
        sigma_inv.data_ptr<float>(),
        opacity.data_ptr<float>(),
        vel.data_ptr<float>(),
        J.data_ptr<float>(),
        cell_ptr.data_ptr<int64_t>(),
        cell_indices.data_ptr<int64_t>(),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(tile_size),
        static_cast<int>(grid_w),
        grad_mu.data_ptr<float>(),
        grad_sigma_inv.data_ptr<float>(),
        grad_opacity.data_ptr<float>(),
        grad_vel.data_ptr<float>(),
        grad_J.data_ptr<float>());

    return {grad_mu, grad_sigma_inv, grad_opacity, grad_vel, grad_J};
}
