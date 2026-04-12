import torch


def add_event_noise(
    events: torch.Tensor,
    H: int,
    W: int,
    noise_ratio: float = 0.0,
    grid_size: int = 8,
    n_hot_pixels: int = 32,
    n_bursts: int = 12,
    burst_std_frac: float = 0.004,
    timestamp_jitter_frac: float = 0.001,
) -> torch.Tensor:
    """
    更接近真实事件相机的噪声注入：
    - 总事件数不变
    - 在空间网格内尽量均匀地选择被污染事件
    - 噪声由四类组成，并尽量等比例分配：
        1) background activity noise   全局随机背景噪声
        2) hot pixel noise             热像素/坏点噪声
        3) burst noise                 时间突发噪声
        4) polarity/timestamp corruption
           其中该类内部再平分为：
             - 极性翻转
             - 轻微时间戳抖动

    参数
    ----
    events : (N, 4) tensor, 格式 [h, w, t, p]，即 yx 顺序
    H, W   : 图像高宽
    noise_ratio : 总噪声比例，定义为被替换/污染事件占总事件数的比例，范围 [0,1]
    grid_size   : 空间均匀采样网格边长
    n_hot_pixels: 热像素个数
    n_bursts    : burst 中心个数
    burst_std_frac : burst 的时间宽度，相对整个时间窗
    timestamp_jitter_frac : 时间戳抖动强度，相对整个时间窗

    返回
    ----
    events_noisy : (N, 4) tensor，与输入等长
    """
    if noise_ratio <= 0.0:
        return events.clone().float()

    out = events.clone().float()
    device = out.device
    N = out.shape[0]

    if N == 0:
        return out

    # ---------- 基本范围 ----------
    t_min = out[:, 2].min()
    t_max = out[:, 2].max()
    t_range = (t_max - t_min).clamp(min=1e-6)

    burst_std = burst_std_frac * t_range
    timestamp_jitter_std = timestamp_jitter_frac * t_range

    # ---------- 计算总噪声数 ----------
    N_noise = int(round(N * noise_ratio))
    N_noise = max(0, min(N_noise, N))
    if N_noise == 0:
        return out

    # ---------- 网格划分：保证空间上尽量均匀选取被污染事件 ----------
    cell_h = torch.clamp((out[:, 0] / H * grid_size).long(), 0, grid_size - 1)
    cell_w = torch.clamp((out[:, 1] / W * grid_size).long(), 0, grid_size - 1)
    cell_id = cell_h * grid_size + cell_w

    num_cells = grid_size * grid_size
    sort_by_cell = torch.argsort(cell_id, stable=True)
    counts = torch.bincount(cell_id, minlength=num_cells)
    offsets = torch.zeros(num_cells + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    # ---------- 每个格子分配噪声数量 ----------
    # 先按比例分配到每个格子，再用余数补齐，保证总数严格等于 N_noise
    target_per_cell = counts.float() * (N_noise / max(N, 1))
    base_per_cell = torch.floor(target_per_cell).long()
    remain = N_noise - int(base_per_cell.sum().item())

    if remain > 0:
        frac = target_per_cell - base_per_cell.float()
        # 只给还有容量的格子补
        can_add = base_per_cell < counts
        frac = torch.where(can_add, frac, torch.full_like(frac, -1.0))
        topk = torch.topk(frac, k=min(remain, int(can_add.sum().item())), largest=True).indices
        base_per_cell[topk] += 1

    # 再防止极端情况下超出格子容量
    base_per_cell = torch.minimum(base_per_cell, counts)

    # ---------- 在每个格子内随机抽取被污染事件 ----------
    selected_list = []
    for c in range(num_cells):
        n_cell = counts[c].item()
        k = base_per_cell[c].item()
        if n_cell == 0 or k == 0:
            continue
        cell_orig_idx = sort_by_cell[offsets[c]:offsets[c + 1]]
        perm = torch.randperm(n_cell, device=device)[:k]
        selected_list.append(cell_orig_idx[perm])

    if len(selected_list) == 0:
        return out

    selected_idx = torch.cat(selected_list, dim=0)

    # 若由于容量限制导致数量不足，额外从剩余事件里补
    if len(selected_idx) < N_noise:
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[selected_idx] = False
        remaining_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        extra_k = min(N_noise - len(selected_idx), len(remaining_idx))
        if extra_k > 0:
            extra_perm = torch.randperm(len(remaining_idx), device=device)[:extra_k]
            selected_idx = torch.cat([selected_idx, remaining_idx[extra_perm]], dim=0)

    # 若超过则截断
    if len(selected_idx) > N_noise:
        perm = torch.randperm(len(selected_idx), device=device)[:N_noise]
        selected_idx = selected_idx[perm]

    # 打乱
    selected_idx = selected_idx[torch.randperm(len(selected_idx), device=device)]

    # ---------- 四大类噪声等比例分配 ----------
    # 1 background
    # 2 hot pixel
    # 3 burst
    # 4 corruption（内部再拆成 polarity flip 和 timestamp jitter）
    n1 = N_noise // 4
    n2 = N_noise // 4
    n3 = N_noise // 4
    n4 = N_noise - n1 - n2 - n3

    idx1 = selected_idx[:n1]
    idx2 = selected_idx[n1:n1 + n2]
    idx3 = selected_idx[n1 + n2:n1 + n2 + n3]
    idx4 = selected_idx[n1 + n2 + n3:]

    # =========================================================
    # 1) Background activity noise
    #    全局随机背景噪声：空间随机，时间随机，极性随机
    # =========================================================
    if len(idx1) > 0:
        n = len(idx1)
        out[idx1, 0] = torch.rand(n, device=device) * (H - 1)  # h
        out[idx1, 1] = torch.rand(n, device=device) * (W - 1)  # w
        out[idx1, 2] = torch.rand(n, device=device) * t_range + t_min
        out[idx1, 3] = torch.randint(0, 2, (n,), device=device).float()  # p in {0,1}

    # =========================================================
    # 2) Hot pixel noise
    #    少数固定像素不断触发：位置固定，时间随机，极性带偏置
    # =========================================================
    if len(idx2) > 0:
        n = len(idx2)

        hot_h = torch.randint(0, H, (n_hot_pixels,), device=device).float()
        hot_w = torch.randint(0, W, (n_hot_pixels,), device=device).float()

        choice = torch.randint(0, n_hot_pixels, (n,), device=device)
        out[idx2, 0] = hot_h[choice]
        out[idx2, 1] = hot_w[choice]
        out[idx2, 2] = torch.rand(n, device=device) * t_range + t_min

        # 为了更像真实坏点，给每个 hot pixel 一个固定偏置极性
        hot_p = torch.randint(0, 2, (n_hot_pixels,), device=device).float()
        out[idx2, 3] = hot_p[choice]

    # =========================================================
    # 3) Burst noise
    #    短时间突发：时间集中，空间随机，单个 burst 内极性有偏置
    # =========================================================
    if len(idx3) > 0:
        n = len(idx3)

        burst_centers = torch.rand(n_bursts, device=device) * t_range + t_min
        choice = torch.randint(0, n_bursts, (n,), device=device)

        burst_t = burst_centers[choice] + torch.randn(n, device=device) * burst_std
        burst_t = burst_t.clamp(t_min, t_max)

        out[idx3, 0] = torch.rand(n, device=device) * (H - 1)
        out[idx3, 1] = torch.rand(n, device=device) * (W - 1)
        out[idx3, 2] = burst_t

        burst_p = torch.randint(0, 2, (n_bursts,), device=device).float()
        out[idx3, 3] = burst_p[choice]

    # =========================================================
    # 4) Corruption
    #    一半做极性翻转，一半做轻微时间戳抖动
    # =========================================================
    if len(idx4) > 0:
        n = len(idx4)
        n_flip = n // 2
        n_jitter = n - n_flip

        idx4 = idx4[torch.randperm(n, device=device)]
        idx_flip = idx4[:n_flip]
        idx_jitter = idx4[n_flip:]

        # 4a) 极性翻转，p ∈ {0,1}
        if len(idx_flip) > 0:
            out[idx_flip, 3] = 1.0 - out[idx_flip, 3]

        # 4b) 轻微时间戳扰动
        if len(idx_jitter) > 0:
            dt = torch.randn(len(idx_jitter), device=device) * timestamp_jitter_std
            out[idx_jitter, 2] = (out[idx_jitter, 2] + dt).clamp(t_min, t_max)

    # ---------- 重新按时间排序 ----------
    sort_idx = torch.argsort(out[:, 2], stable=True)
    out = out[sort_idx]

    return out