import torch

def events_to_voxel_grid(events, num_bins, H, W, t0=None, t1=None):
    """
    Convert events (B,4) to a voxel grid tensor (C,H,W)
    events: tensor of shape [B, 4], columns: [x, y, t, p]
    num_bins: number of temporal bins (C)
    H, W: height and width of the voxel grid
    """

    assert events.shape[1] == 4
    y = events[:, 0].long()
    x = events[:, 1].long()
    t = events[:, 2]
    p = events[:, 3].float()  # polarity: -1 or 1

    # x = x-torch.min(x)
    # y = y-torch.min(y)

    # Normalize timestamps to [0, 1]
    if t0 is None:
        t0 = t.min()
    if t1 is None:
        t1 = t.max()
    t_norm = (t - t0+1e-8) / (t1 - t0 + 1e-8)
    t_idx = (t_norm * (num_bins - 1)).clamp(0, num_bins - 1)

    # Floor and Ceil bins for linear interpolation
    t_low = t_idx.floor().long()
    # t_low = t_idx.floor().long()
    t_high = t_low + 1
    w_high = t_idx - t_low.float()
    w_low = 1.0 - w_high

    # Initialize voxel grid 
    voxel_grid = torch.zeros((num_bins, H, W), dtype=torch.float32, device=events.device)

    # Accumulate using bilinear temporal weightstensor(0.2205, device='cuda:0') tensor(0.0302, device='cuda:0')
    valid_low = t_low < num_bins
    voxel_grid.index_put_((t_low[valid_low], y[valid_low], x[valid_low]),
                          w_low[valid_low], accumulate=True)

    valid_high = t_high < num_bins
    voxel_grid.index_put_((t_high[valid_high], y[valid_high], x[valid_high]), 
                          w_high[valid_high], accumulate=True)
    voxel_grid = voxel_grid[1:-1]
    return voxel_grid, t1-t0



def events_to_est(events, num_bins, H, W, decay_rate=0.5, separate_polarity=False, t0=None, t1=None):
    """
    Convert events (B,4) to a est with optional polarity separation.
    events: tensor of shape [B, 4], columns: [y, x, t, p]
    decay_rate: decay rate for exponential weighting
    t0, t1: optional time normalization range
    
    Returns:
        EST: tensor of shape (2, N, H, W) if separate_polarity=True
                    or (N, H, W) if separate_polarity=False
        duration: t1 - t0
    """
    assert events.shape[1] == 4
    y = events[:, 0].long()
    x = events[:, 1].long()
    t = events[:, 2]
    p = events[:, 3].float()  # polarity: -1 or 1

    # x = x - torch.min(x)
    # y = y - torch.min(y)

    # Normalize timestamps to [0, 1]
    if t0 is None:
        t0 = t.min()
    if t1 is None:
        t1 = t.max()
        
    t_norm = (t - t0 + 1e-8) / (t1 - t0 + 1e-8)
    t_idx = (t_norm * (num_bins - 1)).clamp(0, num_bins - 1)

    # Floor and Ceil bins for linear interpolation
    t_low = t_idx.floor().long()
    t_high = t_low + 1
    
    # 计算权重，同时也是小窗口内（长度为1）时间归一化
    w_l = t_idx - t_low.float()  # 小数部分，后续经过指数衰减，因此不再对应t_high，而是对应t_low的权重
    w_h = 1.0 - w_l               # t_low的权重

    # 计算指数衰减权重
    w_low = w_l * decay_rate * torch.exp(-decay_rate * w_l)
    w_high = w_l * decay_rate * torch.exp(-decay_rate * w_h)
    
    if separate_polarity:
        # 分离正负事件
        pos_mask = p > 0  # 正事件掩码
        neg_mask = p <= 0  # 负事件掩码
        
        # 初始化两个极性通道的体素网格
        est_pos = torch.zeros((num_bins, H, W), dtype=torch.float32, device=events.device)
        est_neg = torch.zeros((num_bins, H, W), dtype=torch.float32, device=events.device)
        
        # 处理正事件
        if torch.any(pos_mask):
            pos_x = x[pos_mask]
            pos_y = y[pos_mask]
            pos_t_low = t_low[pos_mask]
            pos_t_high = t_high[pos_mask]
            pos_w_low = w_low[pos_mask]
            pos_w_high = w_high[pos_mask]
            
            valid_low_pos = pos_t_low < num_bins
            est_pos.index_put_((pos_t_low[valid_low_pos], pos_y[valid_low_pos], pos_x[valid_low_pos]),
                                      pos_w_low[valid_low_pos], accumulate=True)
            
            valid_high_pos = pos_t_high < num_bins
            est_pos.index_put_((pos_t_high[valid_high_pos], pos_y[valid_high_pos], pos_x[valid_high_pos]),
                                      pos_w_high[valid_high_pos], accumulate=True)
        
        # 处理负事件
        if torch.any(neg_mask):
            neg_x = x[neg_mask]
            neg_y = y[neg_mask]
            neg_t_low = t_low[neg_mask]
            neg_t_high = t_high[neg_mask]
            neg_w_low = w_low[neg_mask]
            neg_w_high = w_high[neg_mask]
            
            valid_low_neg = neg_t_low < num_bins
            est_neg.index_put_((neg_t_low[valid_low_neg], neg_y[valid_low_neg], neg_x[valid_low_neg]),
                                      neg_w_low[valid_low_neg], accumulate=True)
            
            valid_high_neg = neg_t_high < num_bins
            est_neg.index_put_((neg_t_high[valid_high_neg], neg_y[valid_high_neg], neg_x[valid_high_neg]),
                                      neg_w_high[valid_high_neg], accumulate=True)
        
        # 移除边界bins
        est_pos = est_pos[1:-1]
        est_neg = est_neg[1:-1]
        
        # 堆叠成正负两个通道 (2, num_bins, H, W)
        est = torch.stack([est_pos, est_neg], dim=0)
        
    else:
        # 不区分极性，所有事件累积到同一个网格
        est = torch.zeros((num_bins, H, W), dtype=torch.float32, device=events.device)
        
        valid_low = t_low < num_bins
        est.index_put_((t_low[valid_low], y[valid_low], x[valid_low]),
                              w_low[valid_low], accumulate=True)
        
        valid_high = t_high < num_bins
        est.index_put_((t_high[valid_high], y[valid_high], x[valid_high]), 
                              w_high[valid_high], accumulate=True)
        
        est = est[1:-1]
    
    return est, t1 - t0
