import math

import torch


GAUSSIAN_VALUE_THRESHOLD = 5e-4


class GaussianSpatialIndexer:
    """
    由高斯响应阈值确定支持区域，再构建 Gaussian 到 tile 的索引。
    """

    def __init__(self, means, scales, rotations, domain_size,
                 tile_size, value_threshold=GAUSSIAN_VALUE_THRESHOLD, max_tile_span=16):
        self.means            = means
        self.scales           = scales
        self.rotations        = rotations.reshape(-1, 1)
        self.H, self.W        = domain_size
        self.tile_size        = tile_size
        self.value_threshold  = float(value_threshold)
        if not 0.0 < self.value_threshold < 1.0:
            raise ValueError("value_threshold must be between 0 and 1")
        self.support_radius   = math.sqrt(-2.0 * math.log(self.value_threshold))
        self.max_tile_span    = max_tile_span
        self.device           = means.device

        self.grid_h    = (self.H + tile_size - 1) // tile_size
        self.grid_w    = (self.W + tile_size - 1) // tile_size
        self.num_tiles = self.grid_h * self.grid_w

        self.cell_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.cell_ptr     = torch.zeros(self.num_tiles + 1, dtype=torch.long,
                                        device=self.device)
        self.build()

    def build(self):
        N      = self.means.shape[0]
        device = self.device
        if N == 0:
            return

        cos_t = torch.cos(self.rotations.squeeze(-1))
        sin_t = torch.sin(self.rotations.squeeze(-1))
        s_h, s_w = self.scales[:, 0], self.scales[:, 1]

        sigma_hh = (s_h * cos_t) ** 2 + (s_w * sin_t) ** 2
        sigma_ww = (s_h * sin_t) ** 2 + (s_w * cos_t) ** 2
        radius_h = self.support_radius * torch.sqrt(sigma_hh)
        radius_w = self.support_radius * torch.sqrt(sigma_ww)

        min_h = self.means[:, 0] - radius_h
        max_h = self.means[:, 0] + radius_h
        min_w = self.means[:, 1] - radius_w
        max_w = self.means[:, 1] + radius_w

        rect_min_row = (min_h / self.tile_size).floor().long().clamp(0, self.grid_h - 1)
        rect_max_row = (max_h / self.tile_size).ceil().long().clamp(0, self.grid_h - 1)
        rect_min_col = (min_w / self.tile_size).floor().long().clamp(0, self.grid_w - 1)
        rect_max_col = (max_w / self.tile_size).ceil().long().clamp(0, self.grid_w - 1)

        rows = rect_max_row - rect_min_row + 1
        cols = rect_max_col - rect_min_col + 1
        valid_mask = (
            (rows <= self.max_tile_span) & (cols <= self.max_tile_span) &
            (rows > 0) & (cols > 0)
        )
        if valid_mask.sum() == 0:
            return

        rect_min_row    = rect_min_row[valid_mask]
        rect_max_row    = rect_max_row[valid_mask]
        rect_min_col    = rect_min_col[valid_mask]
        rect_max_col    = rect_max_col[valid_mask]
        rows            = rows[valid_mask]
        cols            = cols[valid_mask]
        original_indices = torch.where(valid_mask)[0]

        max_h_span = int(rows.max().item())
        max_w_span = int(cols.max().item())

        h_off = torch.arange(max_h_span, device=device)
        w_off = torch.arange(max_w_span, device=device)

        grid_rows = (rect_min_row[:, None, None] + h_off[None, :, None]
                     ).expand(-1, -1, max_w_span)
        grid_cols = (rect_min_col[:, None, None] + w_off[None, None, :]
                     ).expand(-1, max_h_span, -1)

        mask = (
            (grid_rows <= rect_max_row[:, None, None]) &
            (grid_cols <= rect_max_col[:, None, None])
        )
        valid_rows = grid_rows[mask]
        valid_cols = grid_cols[mask]
        valid_gids = original_indices[:, None, None].expand(
            -1, max_h_span, max_w_span)[mask]

        tile_ids = valid_rows * self.grid_w + valid_cols   # [M]

        # 计数排序：O(M + T)
        counts = torch.bincount(tile_ids, minlength=self.num_tiles)
        self.cell_ptr = torch.zeros(self.num_tiles + 1, dtype=torch.long, device=device)
        torch.cumsum(counts, dim=0, out=self.cell_ptr[1:])
        order = torch.argsort(tile_ids, stable=True)
        self.cell_indices = valid_gids[order]


