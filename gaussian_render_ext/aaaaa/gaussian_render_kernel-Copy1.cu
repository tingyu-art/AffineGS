#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

constexpr float kGateCenter = 14.0f;
constexpr float kGateSlope  = -2.0f;
constexpr int   kThreads    = 256;
constexpr int   kChunkSize  = 32;
constexpr int   kWarpSize   = 32;
constexpr int   kWarpsPerBlock = kThreads / kWarpSize;

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ void load_gaussian_to_shared(
    int local_idx,
    const float* __restrict__ mu,
    const float* __restrict__ sigma_inv,
    const float* __restrict__ opacity,
    const float* __restrict__ vel,
    const float* __restrict__ J,
    int64_t gid,
    int64_t* __restrict__ s_gid,
    float* __restrict__ s_mu_h,
    float* __restrict__ s_mu_w,
    float* __restrict__ s_si_hh,
    float* __restrict__ s_si_hw,
    float* __restrict__ s_si_ww,
    float* __restrict__ s_op,
    float* __restrict__ s_vel_h,
    float* __restrict__ s_vel_w,
    float* __restrict__ s_J00,
    float* __restrict__ s_J01,
    float* __restrict__ s_J10,
    float* __restrict__ s_J11) {

    s_gid[local_idx]   = gid;
    s_mu_h[local_idx]  = mu[gid * 2 + 0];
    s_mu_w[local_idx]  = mu[gid * 2 + 1];
    s_si_hh[local_idx] = sigma_inv[gid * 4 + 0];
    s_si_hw[local_idx] = sigma_inv[gid * 4 + 1];
    s_si_ww[local_idx] = sigma_inv[gid * 4 + 3];
    s_op[local_idx]    = opacity[gid];
    s_vel_h[local_idx] = vel[gid * 2 + 0];
    s_vel_w[local_idx] = vel[gid * 2 + 1];
    s_J00[local_idx]   = J[gid * 4 + 0];
    s_J01[local_idx]   = J[gid * 4 + 1];
    s_J10[local_idx]   = J[gid * 4 + 2];
    s_J11[local_idx]   = J[gid * 4 + 3];
}

__global__ void gaussian_render_forward_kernel_tiled(
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
    float* __restrict__ out_num_h,
    float* __restrict__ out_num_w,
    float* __restrict__ out_den) {

    const int tile_id = blockIdx.x;
    const int pix_block_offset = blockIdx.y * blockDim.x;
    const int local_pix = pix_block_offset + threadIdx.x;
    const int pix_per_tile = tile_size * tile_size;

    const int tile_r = tile_id / grid_w;
    const int tile_c = tile_id - tile_r * grid_w;

    bool valid_pix = (local_pix < pix_per_tile);
    int ph = 0;
    int pw = 0;
    int pix = 0;

    if (valid_pix) {
        const int off_h = local_pix / tile_size;
        const int off_w = local_pix - off_h * tile_size;
        ph = tile_r * tile_size + off_h;
        pw = tile_c * tile_size + off_w;
        valid_pix = (ph < H && pw < W);
        if (valid_pix) {
            pix = ph * W + pw;
        }
    }

    const int64_t start = cell_ptr[tile_id];
    const int64_t end   = cell_ptr[tile_id + 1];

    __shared__ int64_t s_gid[kChunkSize];
    __shared__ float s_mu_h[kChunkSize];
    __shared__ float s_mu_w[kChunkSize];
    __shared__ float s_si_hh[kChunkSize];
    __shared__ float s_si_hw[kChunkSize];
    __shared__ float s_si_ww[kChunkSize];
    __shared__ float s_op[kChunkSize];
    __shared__ float s_vel_h[kChunkSize];
    __shared__ float s_vel_w[kChunkSize];
    __shared__ float s_J00[kChunkSize];
    __shared__ float s_J01[kChunkSize];
    __shared__ float s_J10[kChunkSize];
    __shared__ float s_J11[kChunkSize];

    float acc_num_h = 0.0f;
    float acc_num_w = 0.0f;
    float acc_den   = 0.0f;

    const float fph = static_cast<float>(ph);
    const float fpw = static_cast<float>(pw);

    for (int64_t chunk_start = start; chunk_start < end; chunk_start += kChunkSize) {
        const int chunk_count = static_cast<int>(((end - chunk_start) < kChunkSize) ? (end - chunk_start) : kChunkSize);

        if (threadIdx.x < chunk_count) {
            const int64_t gid = cell_idx[chunk_start + threadIdx.x];
            load_gaussian_to_shared(
                threadIdx.x, mu, sigma_inv, opacity, vel, J, gid,
                s_gid, s_mu_h, s_mu_w, s_si_hh, s_si_hw, s_si_ww, s_op,
                s_vel_h, s_vel_w, s_J00, s_J01, s_J10, s_J11);
        }
        __syncthreads();

        if (valid_pix) {
            #pragma unroll
            for (int j = 0; j < kChunkSize; ++j) {
                if (j >= chunk_count) break;

                const float dh = fph - s_mu_h[j];
                const float dw = fpw - s_mu_w[j];

                const float term = dh * dh * s_si_hh[j]
                                 + 2.0f * dh * dw * s_si_hw[j]
                                 + dw * dw * s_si_ww[j];
                const float gate = sigmoidf_fast(kGateSlope * (term - kGateCenter));
                const float base = expf(-0.5f * term) * gate;
                const float w    = base * s_op[j];

                const float v_h = s_vel_h[j] + s_J00[j] * dh + s_J01[j] * dw;
                const float v_w = s_vel_w[j] + s_J10[j] * dh + s_J11[j] * dw;

                acc_num_h += w * v_h;
                acc_num_w += w * v_w;
                acc_den   += w;
            }
        }
        __syncthreads();
    }

    if (valid_pix) {
        out_num_h[pix] = acc_num_h;
        out_num_w[pix] = acc_num_w;
        out_den[pix]   = acc_den;
    }
}

__global__ void gaussian_render_backward_kernel_warp_reduce(
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

    const int tile_id = blockIdx.x;
    const int pix_block_offset = blockIdx.y * blockDim.x;
    const int pix_per_tile = tile_size * tile_size;

    const int tile_r = tile_id / grid_w;
    const int tile_c = tile_id - tile_r * grid_w;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid / kWarpSize;

    __shared__ float s_ph[kThreads];
    __shared__ float s_pw[kThreads];
    __shared__ float s_gNh[kThreads];
    __shared__ float s_gNw[kThreads];
    __shared__ float s_gD[kThreads];
    __shared__ unsigned char s_valid[kThreads];

    const int local_pix = pix_block_offset + tid;
    bool valid_pix = (local_pix < pix_per_tile);
    int ph = 0;
    int pw = 0;
    int pix = 0;

    if (valid_pix) {
        const int off_h = local_pix / tile_size;
        const int off_w = local_pix - off_h * tile_size;
        ph = tile_r * tile_size + off_h;
        pw = tile_c * tile_size + off_w;
        valid_pix = (ph < H && pw < W);
        if (valid_pix) {
            pix = ph * W + pw;
            const float gNh = grad_num_h[pix];
            const float gNw = grad_num_w[pix];
            const float gD  = grad_den[pix];
            const bool has_grad = !(gNh == 0.0f && gNw == 0.0f && gD == 0.0f);
            s_valid[tid] = static_cast<unsigned char>(has_grad);
            s_gNh[tid] = gNh;
            s_gNw[tid] = gNw;
            s_gD[tid]  = gD;
            s_ph[tid] = static_cast<float>(ph);
            s_pw[tid] = static_cast<float>(pw);
        } else {
            s_valid[tid] = 0;
            s_gNh[tid] = 0.0f;
            s_gNw[tid] = 0.0f;
            s_gD[tid]  = 0.0f;
            s_ph[tid] = 0.0f;
            s_pw[tid] = 0.0f;
        }
    } else {
        s_valid[tid] = 0;
        s_gNh[tid] = 0.0f;
        s_gNw[tid] = 0.0f;
        s_gD[tid]  = 0.0f;
        s_ph[tid] = 0.0f;
        s_pw[tid] = 0.0f;
    }
    __syncthreads();

    const int64_t start = cell_ptr[tile_id];
    const int64_t end   = cell_ptr[tile_id + 1];

    __shared__ int64_t s_gid[kChunkSize];
    __shared__ float s_mu_h[kChunkSize];
    __shared__ float s_mu_w[kChunkSize];
    __shared__ float s_si_hh[kChunkSize];
    __shared__ float s_si_hw[kChunkSize];
    __shared__ float s_si_ww[kChunkSize];
    __shared__ float s_op[kChunkSize];
    __shared__ float s_vel_h[kChunkSize];
    __shared__ float s_vel_w[kChunkSize];
    __shared__ float s_J00[kChunkSize];
    __shared__ float s_J01[kChunkSize];
    __shared__ float s_J10[kChunkSize];
    __shared__ float s_J11[kChunkSize];

    for (int64_t chunk_start = start; chunk_start < end; chunk_start += kChunkSize) {
        const int chunk_count = static_cast<int>(((end - chunk_start) < kChunkSize) ? (end - chunk_start) : kChunkSize);

        if (tid < chunk_count) {
            const int64_t gid = cell_idx[chunk_start + tid];
            load_gaussian_to_shared(
                tid, mu, sigma_inv, opacity, vel, J, gid,
                s_gid, s_mu_h, s_mu_w, s_si_hh, s_si_hw, s_si_ww, s_op,
                s_vel_h, s_vel_w, s_J00, s_J01, s_J10, s_J11);
        }
        __syncthreads();

        #pragma unroll
        for (int sub = 0; sub < kChunkSize; sub += kWarpsPerBlock) {
            const int j = sub + warp_id;
            if (j >= chunk_count) continue;

            const float mu_h = s_mu_h[j];
            const float mu_w = s_mu_w[j];
            const float si_hh = s_si_hh[j];
            const float si_hw = s_si_hw[j];
            const float si_ww = s_si_ww[j];
            const float op    = s_op[j];
            const float vel_h = s_vel_h[j];
            const float vel_w = s_vel_w[j];
            const float J00   = s_J00[j];
            const float J01   = s_J01[j];
            const float J10   = s_J10[j];
            const float J11   = s_J11[j];

            float g_mu_h  = 0.0f;
            float g_mu_w  = 0.0f;
            float g_si_hh = 0.0f;
            float g_si_hw = 0.0f;
            float g_si_ww = 0.0f;
            float g_op    = 0.0f;
            float g_vel_h = 0.0f;
            float g_vel_w = 0.0f;
            float g_J00   = 0.0f;
            float g_J01   = 0.0f;
            float g_J10   = 0.0f;
            float g_J11   = 0.0f;

            #pragma unroll
            for (int p = lane; p < kThreads; p += kWarpSize) {
                if (!s_valid[p]) continue;

                const float dh = s_ph[p] - mu_h;
                const float dw = s_pw[p] - mu_w;

                const float term = dh * dh * si_hh
                                 + 2.0f * dh * dw * si_hw
                                 + dw * dw * si_ww;
                const float gate = sigmoidf_fast(kGateSlope * (term - kGateCenter));
                const float base = expf(-0.5f * term) * gate;
                const float w    = base * op;

                const float v_h = vel_h + J00 * dh + J01 * dw;
                const float v_w = vel_w + J10 * dh + J11 * dw;

                const float dL_dw  = s_gNh[p] * v_h + s_gNw[p] * v_w + s_gD[p];
                const float dL_dvh = s_gNh[p] * w;
                const float dL_dvw = s_gNw[p] * w;

                const float dbase_dterm = base * (2.0f * gate - 2.5f);
                const float dwgt_dterm  = op * dbase_dterm;

                const float dterm_ddh = 2.0f * (si_hh * dh + si_hw * dw);
                const float dterm_ddw = 2.0f * (si_hw * dh + si_ww * dw);

                const float dL_ddh = dL_dw * dwgt_dterm * dterm_ddh + dL_dvh * J00 + dL_dvw * J10;
                const float dL_ddw = dL_dw * dwgt_dterm * dterm_ddw + dL_dvh * J01 + dL_dvw * J11;

                g_mu_h  += -dL_ddh;
                g_mu_w  += -dL_ddw;
                g_si_hh += dL_dw * dwgt_dterm * (dh * dh);
                g_si_hw += dL_dw * dwgt_dterm * (2.0f * dh * dw);
                g_si_ww += dL_dw * dwgt_dterm * (dw * dw);
                g_op    += dL_dw * base;
                g_vel_h += dL_dvh;
                g_vel_w += dL_dvw;
                g_J00   += dL_dvh * dh;
                g_J01   += dL_dvh * dw;
                g_J10   += dL_dvw * dh;
                g_J11   += dL_dvw * dw;
            }

            g_mu_h  = warp_sum(g_mu_h);
            g_mu_w  = warp_sum(g_mu_w);
            g_si_hh = warp_sum(g_si_hh);
            g_si_hw = warp_sum(g_si_hw);
            g_si_ww = warp_sum(g_si_ww);
            g_op    = warp_sum(g_op);
            g_vel_h = warp_sum(g_vel_h);
            g_vel_w = warp_sum(g_vel_w);
            g_J00   = warp_sum(g_J00);
            g_J01   = warp_sum(g_J01);
            g_J10   = warp_sum(g_J10);
            g_J11   = warp_sum(g_J11);

            if (lane == 0) {
                const int64_t gid = s_gid[j];
                atomicAdd(&grad_mu[gid * 2 + 0], g_mu_h);
                atomicAdd(&grad_mu[gid * 2 + 1], g_mu_w);

                atomicAdd(&grad_sigma_inv[gid * 4 + 0], g_si_hh);
                atomicAdd(&grad_sigma_inv[gid * 4 + 1], g_si_hw);
                atomicAdd(&grad_sigma_inv[gid * 4 + 3], g_si_ww);

                atomicAdd(&grad_opacity[gid],     g_op);
                atomicAdd(&grad_vel[gid * 2 + 0], g_vel_h);
                atomicAdd(&grad_vel[gid * 2 + 1], g_vel_w);

                atomicAdd(&grad_J[gid * 4 + 0], g_J00);
                atomicAdd(&grad_J[gid * 4 + 1], g_J01);
                atomicAdd(&grad_J[gid * 4 + 2], g_J10);
                atomicAdd(&grad_J[gid * 4 + 3], g_J11);
            }
        }
        __syncthreads();
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

    const int grid_h = static_cast<int>((H + tile_size - 1) / tile_size);
    const int num_tiles = grid_h * static_cast<int>(grid_w);
    const int pix_per_tile = static_cast<int>(tile_size * tile_size);
    const int pix_slices = (pix_per_tile + kThreads - 1) / kThreads;

    dim3 blocks(num_tiles, pix_slices);
    gaussian_render_forward_kernel_tiled<<<blocks, kThreads>>>(
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

    const int grid_h = static_cast<int>((H + tile_size - 1) / tile_size);
    const int num_tiles = grid_h * static_cast<int>(grid_w);
    const int pix_per_tile = static_cast<int>(tile_size * tile_size);
    const int pix_slices = (pix_per_tile + kThreads - 1) / kThreads;

    dim3 blocks(num_tiles, pix_slices);
    gaussian_render_backward_kernel_warp_reduce<<<blocks, kThreads>>>(
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
