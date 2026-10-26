import torch


def _event_time_range(events: torch.Tensor):
    t_min = events[:, 2].min()
    t_max = events[:, 2].max()
    t_range = (t_max - t_min).clamp(min=1e-6)
    return t_min, t_max, t_range


def _sort_by_time(events: torch.Tensor) -> torch.Tensor:
    sort_idx = torch.argsort(events[:, 2], stable=True)
    return events[sort_idx]


def add_event_noise(
    events: torch.Tensor,
    H: int,
    W: int,
    noise_ratio: float = 0.0,
    n_hot_pixels: int = 32,
    n_bursts: int = 12,
    burst_std_frac: float = 0.004,
    background_fraction: float = 0.5,
    hot_pixel_fraction: float = 0.25,
    burst_fraction: float = 0.25,
) -> torch.Tensor:
    """
    Inject background, hot-pixel and burst noise into existing events.

    Select round(N * noise_ratio) distinct rows and replace their event values;
    do not append or remove rows. The returned tensor therefore always has N
    events, sorted by time when noise is injected. The input is not modified.

    noise_ratio is the fraction of existing events selected for corruption and
    must lie in [0, 1]. The noise budget is split between the same three noise
    types (50% background, 25% hot pixels, 25% bursts by default).
    """
    if not 0.0 <= noise_ratio <= 1.0:
        raise ValueError("noise_ratio must be a fraction in [0, 1]")

    clean = events.clone().float()
    N = clean.shape[0]
    if noise_ratio <= 0.0 or N == 0:
        return clean

    device = clean.device
    dtype = clean.dtype
    t_min, t_max, t_range = _event_time_range(clean)

    N_noise = int(round(N * noise_ratio))
    if N_noise <= 0:
        return clean

    fractions = torch.tensor(
        [background_fraction, hot_pixel_fraction, burst_fraction],
        device=device,
        dtype=torch.float32,
    ).clamp_min(0)
    if float(fractions.sum().item()) <= 0:
        fractions = torch.tensor([1.0, 0.0, 0.0], device=device)
    fractions = fractions / fractions.sum()

    counts = torch.floor(fractions * N_noise).long()
    remain = N_noise - int(counts.sum().item())
    if remain > 0:
        frac = fractions * N_noise - counts.float()
        topk = torch.topk(frac, k=remain, largest=True).indices
        counts[topk] += 1

    n_bg, n_hot, n_burst = [int(x.item()) for x in counts]
    noise_parts = []

    # 1) Background activity: random space, random time, random polarity.
    if n_bg > 0:
        bg = torch.empty((n_bg, 4), device=device, dtype=dtype)
        bg[:, 0] = torch.rand(n_bg, device=device, dtype=dtype) * (H - 1)
        bg[:, 1] = torch.rand(n_bg, device=device, dtype=dtype) * (W - 1)
        bg[:, 2] = torch.rand(n_bg, device=device, dtype=dtype) * t_range + t_min
        bg[:, 3] = torch.randint(0, 2, (n_bg,), device=device).to(dtype)
        noise_parts.append(bg)

    # 2) Hot pixels: a few fixed pixels repeatedly emit events.
    if n_hot > 0:
        n_hp = max(1, min(int(n_hot_pixels), n_hot))
        hot_h = torch.randint(0, H, (n_hp,), device=device).to(dtype)
        hot_w = torch.randint(0, W, (n_hp,), device=device).to(dtype)
        hot_p = torch.randint(0, 2, (n_hp,), device=device).to(dtype)
        choice = torch.randint(0, n_hp, (n_hot,), device=device)

        hot = torch.empty((n_hot, 4), device=device, dtype=dtype)
        hot[:, 0] = hot_h[choice]
        hot[:, 1] = hot_w[choice]
        hot[:, 2] = torch.rand(n_hot, device=device, dtype=dtype) * t_range + t_min
        hot[:, 3] = hot_p[choice]
        noise_parts.append(hot)

    # 3) Burst noise: temporally clustered random events.
    if n_burst > 0:
        n_centers = max(1, min(int(n_bursts), n_burst))
        burst_std = burst_std_frac * t_range
        centers = torch.rand(n_centers, device=device, dtype=dtype) * t_range + t_min
        center_p = torch.randint(0, 2, (n_centers,), device=device).to(dtype)
        choice = torch.randint(0, n_centers, (n_burst,), device=device)

        burst = torch.empty((n_burst, 4), device=device, dtype=dtype)
        burst[:, 0] = torch.rand(n_burst, device=device, dtype=dtype) * (H - 1)
        burst[:, 1] = torch.rand(n_burst, device=device, dtype=dtype) * (W - 1)
        burst[:, 2] = (
            centers[choice] + torch.randn(n_burst, device=device, dtype=dtype) * burst_std
        ).clamp(t_min, t_max)
        burst[:, 3] = center_p[choice]
        noise_parts.append(burst)

    if not noise_parts:
        return clean

    # Keep the original noise budget and distributions, but apply them to
    # distinct existing rows instead of increasing the event count.
    noise_events = torch.cat(noise_parts, dim=0)
    noise_events[:, 2].clamp_(t_min, t_max)
    selected_idx = torch.randperm(N, device=device)[:N_noise]
    clean[selected_idx] = noise_events
    return _sort_by_time(clean)


def corrupt_event_noise(
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
    Replacement/corruption noise kept for robustness experiments.

    The output length is unchanged. noise_ratio denotes the fraction of clean
    events that are selected and then replaced or corrupted.
    """
    if noise_ratio <= 0.0:
        return events.clone().float()

    out = events.clone().float()
    device = out.device
    N = out.shape[0]

    if N == 0:
        return out

    t_min, t_max, t_range = _event_time_range(out)

    burst_std = burst_std_frac * t_range
    timestamp_jitter_std = timestamp_jitter_frac * t_range

    N_noise = int(round(N * noise_ratio))
    N_noise = max(0, min(N_noise, N))
    if N_noise == 0:
        return out

    cell_h = torch.clamp((out[:, 0] / H * grid_size).long(), 0, grid_size - 1)
    cell_w = torch.clamp((out[:, 1] / W * grid_size).long(), 0, grid_size - 1)
    cell_id = cell_h * grid_size + cell_w

    num_cells = grid_size * grid_size
    sort_by_cell = torch.argsort(cell_id, stable=True)
    counts = torch.bincount(cell_id, minlength=num_cells)
    offsets = torch.zeros(num_cells + 1, dtype=torch.long, device=device)
    torch.cumsum(counts, dim=0, out=offsets[1:])

    target_per_cell = counts.float() * (N_noise / max(N, 1))
    base_per_cell = torch.floor(target_per_cell).long()
    remain = N_noise - int(base_per_cell.sum().item())

    if remain > 0:
        frac = target_per_cell - base_per_cell.float()
        can_add = base_per_cell < counts
        frac = torch.where(can_add, frac, torch.full_like(frac, -1.0))
        topk = torch.topk(frac, k=min(remain, int(can_add.sum().item())), largest=True).indices
        base_per_cell[topk] += 1

    base_per_cell = torch.minimum(base_per_cell, counts)

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

    if len(selected_idx) < N_noise:
        mask = torch.ones(N, dtype=torch.bool, device=device)
        mask[selected_idx] = False
        remaining_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        extra_k = min(N_noise - len(selected_idx), len(remaining_idx))
        if extra_k > 0:
            extra_perm = torch.randperm(len(remaining_idx), device=device)[:extra_k]
            selected_idx = torch.cat([selected_idx, remaining_idx[extra_perm]], dim=0)

    if len(selected_idx) > N_noise:
        perm = torch.randperm(len(selected_idx), device=device)[:N_noise]
        selected_idx = selected_idx[perm]

    selected_idx = selected_idx[torch.randperm(len(selected_idx), device=device)]

    n1 = N_noise // 4
    n2 = N_noise // 4
    n3 = N_noise // 4
    n4 = N_noise - n1 - n2 - n3

    idx1 = selected_idx[:n1]
    idx2 = selected_idx[n1:n1 + n2]
    idx3 = selected_idx[n1 + n2:n1 + n2 + n3]
    idx4 = selected_idx[n1 + n2 + n3:]

    if len(idx1) > 0:
        n = len(idx1)
        out[idx1, 0] = torch.rand(n, device=device) * (H - 1)
        out[idx1, 1] = torch.rand(n, device=device) * (W - 1)
        out[idx1, 2] = torch.rand(n, device=device) * t_range + t_min
        out[idx1, 3] = torch.randint(0, 2, (n,), device=device).float()

    if len(idx2) > 0:
        n = len(idx2)
        hot_h = torch.randint(0, H, (n_hot_pixels,), device=device).float()
        hot_w = torch.randint(0, W, (n_hot_pixels,), device=device).float()
        choice = torch.randint(0, n_hot_pixels, (n,), device=device)
        out[idx2, 0] = hot_h[choice]
        out[idx2, 1] = hot_w[choice]
        out[idx2, 2] = torch.rand(n, device=device) * t_range + t_min
        hot_p = torch.randint(0, 2, (n_hot_pixels,), device=device).float()
        out[idx2, 3] = hot_p[choice]

    if len(idx3) > 0:
        n = len(idx3)
        burst_centers = torch.rand(n_bursts, device=device) * t_range + t_min
        choice = torch.randint(0, n_bursts, (n,), device=device)
        out[idx3, 0] = torch.rand(n, device=device) * (H - 1)
        out[idx3, 1] = torch.rand(n, device=device) * (W - 1)
        out[idx3, 2] = (burst_centers[choice] + torch.randn(n, device=device) * burst_std).clamp(t_min, t_max)
        burst_p = torch.randint(0, 2, (n_bursts,), device=device).float()
        out[idx3, 3] = burst_p[choice]

    if len(idx4) > 0:
        n = len(idx4)
        n_flip = n // 2
        idx4 = idx4[torch.randperm(n, device=device)]
        idx_flip = idx4[:n_flip]
        idx_jitter = idx4[n_flip:]

        if len(idx_flip) > 0:
            out[idx_flip, 3] = 1.0 - out[idx_flip, 3]

        if len(idx_jitter) > 0:
            dt = torch.randn(len(idx_jitter), device=device) * timestamp_jitter_std
            out[idx_jitter, 2] = (out[idx_jitter, 2] + dt).clamp(t_min, t_max)

    return _sort_by_time(out)
