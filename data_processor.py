import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import tqdm

# ==========================================================================================
# 事件分段提取工具函数
# 输入事件格式为 [N, 4] (h, w, t, p)，保持时间提取逻辑不变
# ==========================================================================================

def extract_segments(events: torch.Tensor, start_time: float, interval: float, duration: float):
    """
    从事件数据中提取时长 duration 相同、初始时间相差 interval 的两个相邻时间段内的事件。
    
    输入:
        events: torch.Tensor [N, 4] (h, w, t, p)
        start_time: 第一段的起始时间
        interval: 两段的起始时间间隔
        duration: 每段持续时间
    输出:
        prev_events, last_events: torch.Tensor [M, 4]
    """
    if not isinstance(events, torch.Tensor):
        events = torch.as_tensor(events)

    # 按时间排序 (第2列是时间 t)
    idx_sorted = torch.argsort(events[:, 2])
    events = events[idx_sorted]

    # 时间归一化 (减去最小时间)
    events = events.clone()  # 避免原地修改
    events[:, 2] -= events[0, 2]
    timestamps = events[:, 2]

    # 提取时间范围
    t1, t2 = start_time, start_time + duration
    t3, t4 = start_time + interval, start_time + interval + duration

    prev_mask = (timestamps >= t1) & (timestamps < t2)
    last_mask = (timestamps >= t3) & (timestamps < t4)

    return events[prev_mask], events[last_mask]

def gaussian_kernel(size, sigma):
    """
    生成一个二维高斯核
    参数：
        size: 高斯核的大小（奇数）
        sigma: 高斯分布的标准差
    返回：
        一个大小为 (size, size) 的二维高斯核
    """
    # 创建从 -size//2 到 size//2 的一维坐标轴
    ax = torch.arange(-(size // 2), size // 2 + 1, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='xy')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

class Events_to_voxelgrid:
    def __init__(self, height, width, device='cuda', eps=1e-6, gaussian_sigma=1.0, gaussian_kernel_size=5):
        self.height = height
        self.width = width
        self.device = device
        self.eps = eps
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_kernel_size = gaussian_kernel_size

        # 预计算高斯核
        if self.gaussian_kernel_size > 0:
            self.gaussian_kernel = gaussian_kernel(self.gaussian_kernel_size, self.gaussian_sigma).to(self.device)
            # 扩展为 [1, 1, k, k] 格式，用于卷积
            self.gaussian_kernel = self.gaussian_kernel.view(1, 1, self.gaussian_kernel_size, self.gaussian_kernel_size)
            self.pad = self.gaussian_kernel_size // 2
        else:
            self.gaussian_kernel = None
            self.pad = 0

    def apply_gaussian_blur(self, voxel_grid):
        """
        对体素网格的每个时间通道应用高斯模糊
        输入: [C, H, W]
        输出: [C, H, W]
        """
        if self.gaussian_kernel is None:
            return voxel_grid
        
        # 增加批次维度 [1, C, H, W]
        voxel_grid_4d = voxel_grid.unsqueeze(0)
        
        # 对每个通道应用高斯模糊
        blurred_grid = torch.zeros_like(voxel_grid_4d)
        for c in range(voxel_grid_4d.shape[1]):
            channel_data = voxel_grid_4d[:, c:c+1, :, :]  # [1, 1, H, W]
            blurred_channel = F.conv2d(channel_data, self.gaussian_kernel, padding=self.pad)
            blurred_grid[:, c, :, :] = blurred_channel.squeeze(1)
        
        # 去掉批次维度 [C, H, W]
        return blurred_grid.squeeze(0)

    def events_to_voxel_grid(self, events, num_bins, normalize=False):
        """
        将事件数据转换为体素网格，最后应用高斯模糊
        """
        H, W = self.height, self.width
        
        if len(events) == 0:
            return torch.zeros((1, num_bins * 2, H, W), device=self.device)

        # 确保事件数据为张量并移至目标设备
        if not isinstance(events, torch.Tensor):
            events = torch.tensor(events, dtype=torch.float32, device=self.device)
        else:
            events = events.to(self.device, dtype=torch.float32)
            
        hs, ws, ts, pols = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        
        # 坐标裁剪与类型转换
        hs = torch.clamp(hs, 0, H - 1).long()
        ws = torch.clamp(ws, 0, W - 1).long()
        pols = (pols > 0).float()

        # 时间归一化与分箱
        t_min, t_max = ts.min(), ts.max()
        delta_t = max(t_max - t_min, 1e-8)
        norm_ts = (ts - t_min) * (num_bins - 1) / delta_t
        tis = torch.floor(norm_ts).long()
        dts = norm_ts - tis
        vals_left = 1.0 - dts
        vals_right = dts
        
        # 初始化体素网格
        voxel_grid = torch.zeros(num_bins * 2, H, W, device=self.device)

        hw = H * W  # 总像素数
        
        # 分别处理两种极性
        for polarity_val, channel_offset in [(1.0, 0), (0.0, num_bins)]:
            mask = (pols == polarity_val)
            if not mask.any():
                continue

            # 提取当前极性的事件
            hs_p = hs[mask]
            ws_p = ws[mask]
            tis_p = tis[mask]
            vals_l = vals_left[mask]
            vals_r = vals_right[mask]

            # 计算空间索引和时间箱索引
            spatial_idx = hs_p * W + ws_p
            bin_l = tis_p + channel_offset
            bin_r = tis_p + 1 + channel_offset

            # 过滤有效索引
            valid_l = (bin_l >= 0) & (bin_l < num_bins * 2)
            valid_r = (bin_r >= 0) & (bin_r < num_bins * 2)

            # 更新体素网格 (三线性插值)
            flat_grid = voxel_grid.view(-1)
            flat_grid.index_add_(0, bin_l[valid_l] * hw + spatial_idx[valid_l], vals_l[valid_l])
            flat_grid.index_add_(0, bin_r[valid_r] * hw + spatial_idx[valid_r], vals_r[valid_r])

        # 应用高斯模糊
        if self.gaussian_kernel is not None:
            voxel_grid = self.apply_gaussian_blur(voxel_grid)

        # 高值裁剪
        flat_grid = voxel_grid.view(num_bins * 2, -1)
        top_vals = torch.quantile(flat_grid, 0.99999, dim=1, keepdim=True)
        voxel_grid = torch.clamp(voxel_grid, max=top_vals.unsqueeze(2))

        # 增加批次维度
        voxel_grid = voxel_grid.unsqueeze(0)

        # 归一化处理
        if normalize:
            mean = voxel_grid.mean(dim=(2, 3), keepdim=True)
            std = voxel_grid.std(dim=(2, 3), keepdim=True)
            voxel_grid = (voxel_grid - mean) / (std + self.eps)

        return voxel_grid

    def segment_to_rep(self, events, num_bins, start_time, interval, duration, normalize=False):
        """
        获取两个时间段的体素网格
        返回: prev_voxel, last_voxel（均为[1, 2*num_bins, H, W]）
        """
        prev_events, last_events = extract_segments(events, start_time, interval, duration)
        prev_voxel = self.events_to_voxel_grid(prev_events, num_bins, normalize)
        last_voxel = self.events_to_voxel_grid(last_events, num_bins, normalize)
        return prev_voxel, last_voxel
    

# ==========================================================================================
# Event Spike Tensor (EST) 表征
# 调整h（竖直）和w（水平）的索引逻辑，与voxel grid保持一致
# ==========================================================================================

class ValueKernel(nn.Module):
    """时间差权重计算核函数"""
    def __init__(self, num_bins, kernel_type='trilinear', exp_tau=1.0, alpha_param=3.0):
        super().__init__()
        self.num_bins = num_bins
        self.kernel_type = kernel_type
        self.exp_tau = exp_tau
        self.alpha_param = alpha_param
        # 预计算权重有效范围（超过此范围权重为0）
        self.limit = 1.0 / (self.num_bins - 1) if self.num_bins > 1 else 1.0

    def forward(self, delta_t):
        """计算时间差对应的权重，支持批量处理"""
        if self.kernel_type == 'trilinear':
            return self._trilinear(delta_t)
        elif self.kernel_type == 'exponential':
            return self._exponential(delta_t)
        elif self.kernel_type == 'alpha':
            return self._alpha(delta_t)
        else:
            raise ValueError(f"不支持的核类型: {self.kernel_type}")

    def _trilinear(self, dt):
        """三线性核：|dt|≤limit时线性衰减，超范围为0"""
        weights = 1 - torch.abs(dt) / self.limit
        return torch.clamp(weights, 0.0, 1.0)

    def _exponential(self, dt):
        """指数核：|dt|≤limit时指数衰减，超范围为0"""
        weights = torch.exp(-torch.abs(dt) / self.exp_tau)
        valid_mask = (torch.abs(dt) <= self.limit).float()
        return weights * valid_mask

    def _alpha(self, dt):
        """Alpha线性核：|dt|≤limit时Alpha衰减，超范围为0"""
        weights = 1 - self.alpha_param * torch.abs(dt)
        valid_mask = (torch.abs(dt) <= self.limit).float()
        return torch.clamp(weights, 0.0, 1.0) * valid_mask


class ESTQuantizer(nn.Module):
    """事件时空(EST)表征量化器，带高斯扩散功能"""
    def __init__(self, voxel_dim=(9, 180, 240), kernel_type='trilinear',
                 exp_tau=1.0, alpha_param=3.0, gaussian_sigma=1.0, 
                 gaussian_kernel_size=5, use_polarity=True):
        """
        Args:
            voxel_dim: (C, H, W) 时间分箱数, 高度, 宽度
            kernel_type: 核函数类型
            exp_tau: 指数核的衰减系数
            alpha_param: Alpha核的参数
            gaussian_sigma: 高斯扩散的标准差
            gaussian_kernel_size: 高斯核大小
            use_polarity: 是否区分极性 (True: 输出2C通道, False: 输出C通道)
        """
        super().__init__()
        self.C, self.H, self.W = voxel_dim  # C:时间分箱数，H:高度，W:宽度
        self.use_polarity = use_polarity  # 是否区分极性
        
        self.kernel = ValueKernel(
            num_bins=self.C,
            kernel_type=kernel_type,
            exp_tau=exp_tau,
            alpha_param=alpha_param
        )
        
        # 预计算时间分箱中心（范围[0,1]，与归一化时间戳对齐）
        self.register_buffer(
            't_centers',
            torch.linspace(0, 1, self.C) if self.C > 1 else torch.tensor([0.5])
        )
        
        # 高斯扩散参数初始化
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_kernel_size = gaussian_kernel_size
        if self.gaussian_kernel_size > 0:
            # 预计算高斯核并注册为缓冲区（不参与训练）
            self.register_buffer(
                'gaussian_kernel',
                self._create_gaussian_kernel(self.gaussian_kernel_size, self.gaussian_sigma)
            )
            self.gaussian_pad = self.gaussian_kernel_size // 2
            # 扩展维度为[1, 1, k, k]，方便后续卷积操作
            self.gaussian_kernel = self.gaussian_kernel.view(1, 1, self.gaussian_kernel_size, self.gaussian_kernel_size)
        else:
            self.gaussian_kernel = None
            self.gaussian_pad = 0
            
    def _create_gaussian_kernel(self, kernel_size, sigma):
        """创建高斯核"""
        kernel = torch.tensor([
            [np.exp(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2) / (2 * sigma**2))
             for x in range(kernel_size)] 
            for y in range(kernel_size)
        ])
        kernel = kernel / kernel.sum()
        return kernel

    def _apply_gaussian_diffusion(self, grid):
        """对事件网格应用高斯扩散（向量化实现）"""
        if self.gaussian_kernel is None:
            return grid  # 禁用高斯功能时直接返回
        
        # 1. 将1D空间索引重塑为2D网格 [C, H, W]
        grid_2d = grid.view(self.C, self.H, self.W)
        # 2. 边缘填充处理边界
        padded_grid = torch.nn.functional.pad(
            grid_2d, 
            (self.gaussian_pad, self.gaussian_pad, self.gaussian_pad, self.gaussian_pad),
            mode='constant', value=0
        )
        # 3. 使用分组卷积实现向量化高斯扩散
        gaussian_kernel_expanded = self.gaussian_kernel.expand(self.C, 1, -1, -1)
        diffused_grid = torch.nn.functional.conv2d(
            padded_grid.unsqueeze(0),  # 增加批次维度 [1, C, H+p, W+p]
            gaussian_kernel_expanded,  # 卷积核 [C, 1, k, k]
            groups=self.C,  # 每个时间分箱独立卷积
            padding=0
        )
        # 4. 重塑回1D空间索引格式 [C, H*W]
        return diffused_grid.squeeze(0).view(self.C, self.H * self.W)

    def forward(self, events):
        """将事件转换为EST表征"""
        if events.numel() == 0:
            # 根据是否区分极性返回不同维度的零张量
            output_channels = 2 * self.C if self.use_polarity else self.C
            return torch.zeros((1, output_channels, self.H, self.W), device=events.device)
        
        # 1. 分离事件各维度
        # 注意：这里假设输入格式为 [N, 4]，列顺序为 [x, y, t, p] 或 [h, w, t, p]
        h, w, t, p = events.t()
        
        # 2. 空间坐标裁剪（防止超出网格范围）
        h = h.clamp(0, self.H - 1).long()
        w = w.clamp(0, self.W - 1).long()
        
        # 3. 时间归一化到[0,1]范围
        t_min, t_max = t.min(), t.max()
        t_norm = (t - t_min) / (t_max - t_min + 1e-6)  # 加小值避免除零
        
        # 4. 计算空间基础索引（2D→1D）
        base_idx = h * self.W + w
        
        if self.use_polarity:
            # 5. 初始化正负极性网格
            grid_pos = events.new_zeros((self.C, self.H * self.W))  # 正极性(p=1)
            grid_neg = events.new_zeros((self.C, self.H * self.W))  # 负极性(p=0)
            
            # 6. 批量计算时间差与权重
            t_expanded = t_norm.unsqueeze(1)  # [N, 1]
            centers_expanded = self.t_centers.unsqueeze(0)  # [1, C]
            dt = t_expanded - centers_expanded  # [N, C]：每个事件对每个分箱的时间差
            weights = self.kernel(dt)  # [N, C]：计算权重
            
            # 7. 计算贡献值（数值×权重）
            contributions = t_norm.unsqueeze(1) * weights  # [N, C]
            
            # 8. 正负极性贡献累积
            # 正极性事件处理
            pos_mask = (p == 1).unsqueeze(1).float()
            pos_contrib = contributions * pos_mask
            for c in range(self.C):
                c_mask = (pos_contrib[:, c] > 1e-8)  # 过滤微小值（浮点误差）
                if c_mask.sum() == 0:
                    continue
                grid_pos[c].scatter_add_(0, base_idx[c_mask], pos_contrib[c_mask, c])
            
            # 负极性事件处理
            neg_mask = (p == 0).unsqueeze(1).float()
            neg_contrib = contributions * neg_mask
            for c in range(self.C):
                c_mask = (neg_contrib[:, c] > 1e-8)
                if c_mask.sum() == 0:
                    continue
                grid_neg[c].scatter_add_(0, base_idx[c_mask], neg_contrib[c_mask, c])
            
            # 9. 应用高斯扩散
            grid_pos = self._apply_gaussian_diffusion(grid_pos)
            grid_neg = self._apply_gaussian_diffusion(grid_neg)
            
            # 10. 重塑并拼接正负极性
            est = torch.cat([
                grid_pos.view(self.C, self.H, self.W),
                grid_neg.view(self.C, self.H, self.W)
            ], dim=0).unsqueeze(0)  # 最终形状: [1, 2C, H, W]
            
        else:
            # 不区分极性，所有事件合并到同一个网格
            grid_all = events.new_zeros((self.C, self.H * self.W))
            
            # 6. 批量计算时间差与权重
            t_expanded = t_norm.unsqueeze(1)  # [N, 1]
            centers_expanded = self.t_centers.unsqueeze(0)  # [1, C]
            dt = t_expanded - centers_expanded  # [N, C]：每个事件对每个分箱的时间差
            weights = self.kernel(dt)  # [N, C]：计算权重
            
            # 7. 计算贡献值（数值×权重）
            contributions = t_norm.unsqueeze(1) * weights  # [N, C]
            
            # 8. 所有事件贡献累积（不区分极性）
            for c in range(self.C):
                c_mask = (contributions[:, c] > 1e-8)  # 过滤微小值
                if c_mask.sum() == 0:
                    continue
                grid_all[c].scatter_add_(0, base_idx[c_mask], contributions[c_mask, c])
            
            # 9. 应用高斯扩散
            grid_all = self._apply_gaussian_diffusion(grid_all)
            
            # 10. 重塑为EST表征
            est = grid_all.view(self.C, self.H, self.W).unsqueeze(0)  # 最终形状: [1, C, H, W]
        
        return est


class Events_to_EST:
    """事件转EST表征的工具类"""
    def __init__(self, height, width, device='cuda', kernel_type='trilinear',
                 exp_tau=1.0, alpha_param=3.0, gaussian_sigma=1.0, 
                 gaussian_kernel_size=5, use_polarity=True):
        self.height = height
        self.width = width
        self.device = device
        self.kernel_type = kernel_type
        self.exp_tau = exp_tau
        self.alpha_param = alpha_param
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_kernel_size = gaussian_kernel_size
        self.use_polarity = use_polarity  # 是否区分极性
        
        self.quantizer = None
        self.current_num_bins = None  # 跟踪当前分箱数，变化时重新初始化

    def to_est(self, events, num_bins):
        """将事件转换为EST表征"""
        if len(events) == 0:
            # 根据是否区分极性返回不同维度的零张量
            output_channels = 2 * num_bins if self.use_polarity else num_bins
            return torch.zeros((1, output_channels, self.height, self.width), device=self.device)
        
        # 分箱数变化或未初始化时，创建新的量化器
        if self.quantizer is None or self.current_num_bins != num_bins:
            self.quantizer = ESTQuantizer(
                voxel_dim=(num_bins, self.height, self.width),
                kernel_type=self.kernel_type,
                exp_tau=self.exp_tau,
                alpha_param=self.alpha_param,
                gaussian_sigma=self.gaussian_sigma,
                gaussian_kernel_size=self.gaussian_kernel_size,
                use_polarity=self.use_polarity  # 传递极性设置
            ).to(self.device)
            self.current_num_bins = num_bins
        
        # 确保事件为张量并移至目标设备
        if not isinstance(events, torch.Tensor):
            events = torch.tensor(events, dtype=torch.float32, device=self.device)
        else:
            events = events.clone().detach().to(dtype=torch.float32, device=self.device)
        
        return self.quantizer(events)

    def segment_to_rep(self, events, num_bins, start_time, interval, duration):
        """提取两个时间段的EST表征"""
        # 这里假设extract_segments函数已经定义
        prev_events, last_events = extract_segments(events, start_time, interval, duration)
        return self.to_est(prev_events, num_bins), self.to_est(last_events, num_bins)