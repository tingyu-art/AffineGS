#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace {

// Truncate the Gaussian response before multiplying by opacity.
constexpr float kGaussianValueThreshold = 5e-4f;

// Keep launch/layout tuned independently from the value threshold.
constexpr int kThreadsFwd   = 512;   // 32x32 tile -> 1024 pixels -> 2 slices per tile
constexpr int kThreadsBwd   = 256;   // good balance for warp-reduce backward
constexpr int kChunkFwd     = 64;    // fewer chunk iterations / syncs
constexpr int kChunkBwd     = 64;
constexpr int kWarpSize     = 32;
constexpr int kWarpsPerBwd  = kThreadsBwd / kWarpSize;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

struct GaussChunkFwd {
    float2 mu;
    float4 si;   // [hh, hw, wh, ww]
    float  op;
    float2 vel;
    float4 J;    // [00, 01, 10, 11]
};

struct GaussChunkBwd {
    int64_t gid;
    float2 mu;
    float4 si;
    float  op;
    float2 vel;
    float4 J;
};

__device__ __forceinline__ GaussChunkFwd load_gauss_fwd(
    const float2* __restrict__ mu2,
    const float4* __restrict__ sigma4,
    const float*  __restrict__ opacity,
    const float2* __restrict__ vel2,
    const float4* __restrict__ J4,
    int64_t gid) {

    GaussChunkFwd g;
    g.mu  = mu2[gid];
    g.si  = sigma4[gid];
    g.op  = opacity[gid];
    g.vel = vel2[gid];
    g.J   = J4[gid];
    return g;
}

__device__ __forceinline__ GaussChunkBwd load_gauss_bwd(
    const float2* __restrict__ mu2,
    const float4* __restrict__ sigma4,
    const float*  __restrict__ opacity,
    const float2* __restrict__ vel2,
    const float4* __restrict__ J4,
    int64_t gid) {

    GaussChunkBwd g;
    g.gid = gid;
    g.mu  = mu2[gid];
    g.si  = sigma4[gid];
    g.op  = opacity[gid];
    g.vel = vel2[gid];
    g.J   = J4[gid];
    return g;
}

__global__ __launch_bounds__(kThreadsFwd, 1)
void gaussian_render_forward_kernel_tiled_exact(
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

    const int64_t start = cell_ptr[tile_id];
    const int64_t end   = cell_ptr[tile_id + 1];
    if (start == end) {
        return; // outputs are already zero-initialized
    }

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

    const float fph = static_cast<float>(ph);
    const float fpw = static_cast<float>(pw);

    const float2* __restrict__ mu2    = reinterpret_cast<const float2*>(mu);
    const float4* __restrict__ sigma4 = reinterpret_cast<const float4*>(sigma_inv);
    const float2* __restrict__ vel2   = reinterpret_cast<const float2*>(vel);
    const float4* __restrict__ J4     = reinterpret_cast<const float4*>(J);

    __shared__ GaussChunkFwd s_g[kChunkFwd];

    float acc_num_h = 0.0f;
    float acc_num_w = 0.0f;
    float acc_den   = 0.0f;

    for (int64_t chunk_start = start; chunk_start < end; chunk_start += kChunkFwd) {
        const int chunk_count = static_cast<int>(((end - chunk_start) < kChunkFwd) ? (end - chunk_start) : kChunkFwd);

        for (int idx = threadIdx.x; idx < chunk_count; idx += blockDim.x) {
            const int64_t gid = cell_idx[chunk_start + idx];
            s_g[idx] = load_gauss_fwd(mu2, sigma4, opacity, vel2, J4, gid);
        }
        __syncthreads();

        if (valid_pix) {
            #pragma unroll 4
            for (int j = 0; j < kChunkFwd; ++j) {
                if (j >= chunk_count) break;

                const GaussChunkFwd gj = s_g[j];
                const float dh = fph - gj.mu.x;
                const float dw = fpw - gj.mu.y;

                const float term = dh * dh * gj.si.x
                                 + 2.0f * dh * dw * gj.si.y
                                 + dw * dw * gj.si.w;
                const float base = expf(-0.5f * term);
                if (base < kGaussianValueThreshold) continue;
                const float w    = base * gj.op;

                const float v_h = gj.vel.x + gj.J.x * dh + gj.J.y * dw;
                const float v_w = gj.vel.y + gj.J.z * dh + gj.J.w * dw;

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

__global__ __launch_bounds__(kThreadsBwd, 2)
void gaussian_render_backward_kernel_warp_reduce_exact(
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

    const int64_t start = cell_ptr[tile_id];
    const int64_t end   = cell_ptr[tile_id + 1];
    if (start == end) {
        return;
    }

    const int tile_r = tile_id / grid_w;
    const int tile_c = tile_id - tile_r * grid_w;

    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp_id = tid / kWarpSize;

    __shared__ float s_ph[kThreadsBwd];
    __shared__ float s_pw[kThreadsBwd];
    __shared__ float s_gNh[kThreadsBwd];
    __shared__ float s_gNw[kThreadsBwd];
    __shared__ float s_gD[kThreadsBwd];
    __shared__ unsigned char s_valid[kThreadsBwd];
    __shared__ int s_any_grad;

    if (tid == 0) s_any_grad = 0;
    __syncthreads();

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
            s_ph[tid]  = static_cast<float>(ph);
            s_pw[tid]  = static_cast<float>(pw);
            if (has_grad) {
                atomicExch(&s_any_grad, 1);
            }
        } else {
            s_valid[tid] = 0;
            s_gNh[tid] = 0.0f;
            s_gNw[tid] = 0.0f;
            s_gD[tid]  = 0.0f;
            s_ph[tid]  = 0.0f;
            s_pw[tid]  = 0.0f;
        }
    } else {
        s_valid[tid] = 0;
        s_gNh[tid] = 0.0f;
        s_gNw[tid] = 0.0f;
        s_gD[tid]  = 0.0f;
        s_ph[tid]  = 0.0f;
        s_pw[tid]  = 0.0f;
    }
    __syncthreads();

    if (!s_any_grad) {
        return;
    }

    const float2* __restrict__ mu2    = reinterpret_cast<const float2*>(mu);
    const float4* __restrict__ sigma4 = reinterpret_cast<const float4*>(sigma_inv);
    const float2* __restrict__ vel2   = reinterpret_cast<const float2*>(vel);
    const float4* __restrict__ J4     = reinterpret_cast<const float4*>(J);

    __shared__ GaussChunkBwd s_g[kChunkBwd];

    for (int64_t chunk_start = start; chunk_start < end; chunk_start += kChunkBwd) {
        const int chunk_count = static_cast<int>(((end - chunk_start) < kChunkBwd) ? (end - chunk_start) : kChunkBwd);

        if (tid < chunk_count) {
            const int64_t gid = cell_idx[chunk_start + tid];
            s_g[tid] = load_gauss_bwd(mu2, sigma4, opacity, vel2, J4, gid);
        }
        __syncthreads();

        #pragma unroll 4
        for (int sub = 0; sub < kChunkBwd; sub += kWarpsPerBwd) {
            const int j = sub + warp_id;
            if (j >= chunk_count) continue;

            const GaussChunkBwd gj = s_g[j];

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

            for (int p = lane; p < kThreadsBwd; p += kWarpSize) {
                if (!s_valid[p]) continue;

                const float dh = s_ph[p] - gj.mu.x;
                const float dw = s_pw[p] - gj.mu.y;

                const float term = dh * dh * gj.si.x
                                 + 2.0f * dh * dw * gj.si.y
                                 + dw * dw * gj.si.w;
                const float base = expf(-0.5f * term);
                if (base < kGaussianValueThreshold) continue;
                const float w    = base * gj.op;

                const float v_h = gj.vel.x + gj.J.x * dh + gj.J.y * dw;
                const float v_w = gj.vel.y + gj.J.z * dh + gj.J.w * dw;

                const float dL_dw  = s_gNh[p] * v_h + s_gNw[p] * v_w + s_gD[p];
                const float dL_dvh = s_gNh[p] * w;
                const float dL_dvw = s_gNw[p] * w;

                const float dbase_dterm = -0.5f * base;
                const float dwgt_dterm  = gj.op * dbase_dterm;

                const float dterm_ddh = 2.0f * (gj.si.x * dh + gj.si.y * dw);
                const float dterm_ddw = 2.0f * (gj.si.y * dh + gj.si.w * dw);

                const float dL_ddh = dL_dw * dwgt_dterm * dterm_ddh + dL_dvh * gj.J.x + dL_dvw * gj.J.z;
                const float dL_ddw = dL_dw * dwgt_dterm * dterm_ddw + dL_dvh * gj.J.y + dL_dvw * gj.J.w;

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
                const int64_t gid = gj.gid;
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
    const int pix_slices = (pix_per_tile + kThreadsFwd - 1) / kThreadsFwd;

    dim3 blocks(num_tiles, pix_slices);
    gaussian_render_forward_kernel_tiled_exact<<<blocks, kThreadsFwd>>>(
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
    const int pix_slices = (pix_per_tile + kThreadsBwd - 1) / kThreadsBwd;

    dim3 blocks(num_tiles, pix_slices);
    gaussian_render_backward_kernel_warp_reduce_exact<<<blocks, kThreadsBwd>>>(
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
