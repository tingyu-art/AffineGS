#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float clamp_float(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__ int clamp_int(int x, int lo, int hi) {
    return max(lo, min(x, hi));
}

__device__ __forceinline__ float read_flow(
    const float* flow, int c, int H, int W, int h, int w) {
    return flow[(c * H + h) * W + w];
}

__device__ __forceinline__ void bilinear_sample_border(
    const float* flow,
    int H, int W,
    float h, float w,
    float& vel_h,
    float& vel_w,
    int& h0, int& h1, int& w0, int& w1,
    float& w00, float& w01, float& w10, float& w11) {

    float hc = clamp_float(h, 0.0f, float(H - 1));
    float wc = clamp_float(w, 0.0f, float(W - 1));

    h0 = int(floorf(hc));
    w0 = int(floorf(wc));
    h1 = min(h0 + 1, H - 1);
    w1 = min(w0 + 1, W - 1);

    float ah = hc - float(h0);
    float aw = wc - float(w0);

    w00 = (1.0f - ah) * (1.0f - aw);
    w01 = (1.0f - ah) * aw;
    w10 = ah * (1.0f - aw);
    w11 = ah * aw;

    vel_h = 0.0f;
    vel_w = 0.0f;

    vel_h += w00 * read_flow(flow, 0, H, W, h0, w0);
    vel_h += w01 * read_flow(flow, 0, H, W, h0, w1);
    vel_h += w10 * read_flow(flow, 0, H, W, h1, w0);
    vel_h += w11 * read_flow(flow, 0, H, W, h1, w1);

    vel_w += w00 * read_flow(flow, 1, H, W, h0, w0);
    vel_w += w01 * read_flow(flow, 1, H, W, h0, w1);
    vel_w += w10 * read_flow(flow, 1, H, W, h1, w0);
    vel_w += w11 * read_flow(flow, 1, H, W, h1, w1);
}

__global__ void fused_forward_kernel(
    const float* __restrict__ events,   // [N,4]
    const float* __restrict__ flow,     // [2,H,W]
    float* __restrict__ iwe,            // [2,H,W]
    int N, int H, int W,
    float t_ref,
    float inv_scale,
    int radius) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float h = events[idx * 4 + 0];
    const float w = events[idx * 4 + 1];
    const float t = events[idx * 4 + 2];
    const float p = events[idx * 4 + 3];

    float vel_h, vel_w;
    int h0, h1, w0, w1;
    float bw00, bw01, bw10, bw11;
    bilinear_sample_border(
        flow, H, W, h, w,
        vel_h, vel_w, h0, h1, w0, w1,
        bw00, bw01, bw10, bw11
    );

    const float dt = t_ref - t;
    const float h_warp = h + vel_h * dt;
    const float w_warp = w + vel_w * dt;

    constexpr float margin = 1.0f;
    constexpr float beta   = 5.0f;

    const float s1 = sigmoid_fast((h_warp + margin) * beta);
    const float s2 = sigmoid_fast((float(H - 1) + margin - h_warp) * beta);
    const float s3 = sigmoid_fast((w_warp + margin) * beta);
    const float s4 = sigmoid_fast((float(W - 1) + margin - w_warp) * beta);
    const float boundary_weight = s1 * s2 * s3 * s4;

    if (boundary_weight <= 1e-4f) return;

    const int ch = (p > 0.5f) ? 1 : 0;

    const float h_floor_f = floorf(h_warp);
    const float w_floor_f = floorf(w_warp);
    const int h_floor_i = int(h_floor_f);
    const int w_floor_i = int(w_floor_f);

    const float frac_h = h_warp - h_floor_f;
    const float frac_w = w_warp - w_floor_f;

    for (int dh = -radius; dh <= radius; ++dh) {
        for (int dw = -radius; dw <= radius; ++dw) {
            const int ph = clamp_int(h_floor_i + dh, 0, H - 1);
            const int pw = clamp_int(w_floor_i + dw, 0, W - 1);

            const float d_h = frac_h - float(dh);
            const float d_w = frac_w - float(dw);
            const float dist2 = d_h * d_h + d_w * d_w;
            const float g = expf(dist2 * inv_scale) * boundary_weight;

            atomicAdd(&iwe[(ch * H + ph) * W + pw], g);
        }
    }
}

__global__ void fused_backward_kernel(
    const float* __restrict__ grad_iwe, // [2,H,W]
    const float* __restrict__ events,   // [N,4]
    const float* __restrict__ flow,     // [2,H,W]
    float* __restrict__ grad_flow,      // [2,H,W]
    int N, int H, int W,
    float t_ref,
    float inv_scale,
    int radius) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float h = events[idx * 4 + 0];
    const float w = events[idx * 4 + 1];
    const float t = events[idx * 4 + 2];
    const float p = events[idx * 4 + 3];

    float vel_h, vel_w;
    int h0, h1, w0, w1;
    float bw00, bw01, bw10, bw11;
    bilinear_sample_border(
        flow, H, W, h, w,
        vel_h, vel_w, h0, h1, w0, w1,
        bw00, bw01, bw10, bw11
    );

    const float dt = t_ref - t;
    const float h_warp = h + vel_h * dt;
    const float w_warp = w + vel_w * dt;

    constexpr float margin = 1.0f;
    constexpr float beta   = 5.0f;

    const float s1 = sigmoid_fast((h_warp + margin) * beta);
    const float s2 = sigmoid_fast((float(H - 1) + margin - h_warp) * beta);
    const float s3 = sigmoid_fast((w_warp + margin) * beta);
    const float s4 = sigmoid_fast((float(W - 1) + margin - w_warp) * beta);
    const float boundary_weight = s1 * s2 * s3 * s4;

    if (boundary_weight <= 1e-4f) return;

    const float ds1_dh = beta * s1 * (1.0f - s1);
    const float ds2_dh = -beta * s2 * (1.0f - s2);
    const float ds3_dw = beta * s3 * (1.0f - s3);
    const float ds4_dw = -beta * s4 * (1.0f - s4);

    const float dbw_dh = ds1_dh * s2 * s3 * s4 + s1 * ds2_dh * s3 * s4;
    const float dbw_dw = s1 * s2 * ds3_dw * s4 + s1 * s2 * s3 * ds4_dw;

    const int ch = (p > 0.5f) ? 1 : 0;

    const float h_floor_f = floorf(h_warp);
    const float w_floor_f = floorf(w_warp);
    const int h_floor_i = int(h_floor_f);
    const int w_floor_i = int(w_floor_f);

    const float frac_h = h_warp - h_floor_f;
    const float frac_w = w_warp - w_floor_f;

    float grad_h_warp = 0.0f;
    float grad_w_warp = 0.0f;

    for (int dh = -radius; dh <= radius; ++dh) {
        for (int dw = -radius; dw <= radius; ++dw) {
            const int ph = clamp_int(h_floor_i + dh, 0, H - 1);
            const int pw = clamp_int(w_floor_i + dw, 0, W - 1);

            const float upstream = grad_iwe[(ch * H + ph) * W + pw];

            const float d_h = frac_h - float(dh);
            const float d_w = frac_w - float(dw);
            const float dist2 = d_h * d_h + d_w * d_w;

            const float exp_term = expf(dist2 * inv_scale);
            const float g = exp_term * boundary_weight;

            const float dg_dh =
                exp_term * dbw_dh +
                g * (2.0f * d_h * inv_scale);

            const float dg_dw =
                exp_term * dbw_dw +
                g * (2.0f * d_w * inv_scale);

            grad_h_warp += upstream * dg_dh;
            grad_w_warp += upstream * dg_dw;
        }
    }

    const float grad_vel_h = grad_h_warp * dt;
    const float grad_vel_w = grad_w_warp * dt;

    atomicAdd(&grad_flow[(0 * H + h0) * W + w0], grad_vel_h * bw00);
    atomicAdd(&grad_flow[(0 * H + h0) * W + w1], grad_vel_h * bw01);
    atomicAdd(&grad_flow[(0 * H + h1) * W + w0], grad_vel_h * bw10);
    atomicAdd(&grad_flow[(0 * H + h1) * W + w1], grad_vel_h * bw11);

    atomicAdd(&grad_flow[(1 * H + h0) * W + w0], grad_vel_w * bw00);
    atomicAdd(&grad_flow[(1 * H + h0) * W + w1], grad_vel_w * bw01);
    atomicAdd(&grad_flow[(1 * H + h1) * W + w0], grad_vel_w * bw10);
    atomicAdd(&grad_flow[(1 * H + h1) * W + w1], grad_vel_w * bw11);
}

} // namespace

torch::Tensor fused_warp_iwe_forward_cuda(
    torch::Tensor events,
    torch::Tensor flow,
    int64_t H,
    int64_t W,
    double t_ref,
    double sigma,
    int64_t kernel_size) {

    const int N = int(events.size(0));
    auto iwe = torch::zeros({2, H, W}, events.options());

    const int radius = int(kernel_size / 2);
    const float inv_scale = float(-1.0 / (2.0 * sigma * sigma));

    const int blocks = (N + THREADS - 1) / THREADS;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    fused_forward_kernel<<<blocks, THREADS, 0, stream>>>(
        events.data_ptr<float>(),
        flow.data_ptr<float>(),
        iwe.data_ptr<float>(),
        N, int(H), int(W),
        float(t_ref),
        inv_scale,
        radius
    );

    return iwe;
}

torch::Tensor fused_warp_iwe_backward_cuda(
    torch::Tensor grad_iwe,
    torch::Tensor events,
    torch::Tensor flow,
    int64_t H,
    int64_t W,
    double t_ref,
    double sigma,
    int64_t kernel_size) {

    const int N = int(events.size(0));
    auto grad_flow = torch::zeros_like(flow);

    const int radius = int(kernel_size / 2);
    const float inv_scale = float(-1.0 / (2.0 * sigma * sigma));

    const int blocks = (N + THREADS - 1) / THREADS;
    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

    fused_backward_kernel<<<blocks, THREADS, 0, stream>>>(
        grad_iwe.data_ptr<float>(),
        events.data_ptr<float>(),
        flow.data_ptr<float>(),
        grad_flow.data_ptr<float>(),
        N, int(H), int(W),
        float(t_ref),
        inv_scale,
        radius
    );

    return grad_flow;
}