import cv2
import time
import torch
import numpy as np
import torch.nn as nn
import concurrent.futures
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts


# =======================================================================================
#                        method of Neural Implicit Representation(NIR)
# =======================================================================================

class NIR_Base:
    def __init__(self, config=None):
        """
        初始化事件光流估计网络
        config (dict): 配置字典，可选参数见DEFAULT_CONFIG
        """
        self.default_config = {
                'height': 256,
                'width': 256,
                'random_seed': 100,      # 空间编码随机种子
                'encoding_dim': 32,      # 空间编码维度（频率数）
                'beta': 700,             # 空间嵌入缩放因子
                'epochs': 10000,         # 训练轮数
                'lr': 0.001,             # 初始学习率
                'attenuation_factor': 0.8,     # 学习率衰减因子
                'attenuation_step': 2000,      # 学习率衰减步长
                'deform_loss_type': 'l1',      # 形变损失类型 (l1, l2, trick)
                'trick_eps':0.001,
                'trick_alpha':0.5,
                'smooth_loss_type': 'l2',      # 平滑损失范数类型 (l1, l2)
                'smooth_param': 0.01,           # 平滑损失权重
                'verbose': True,               # 可视化开关
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        # 合并用户配置和默认配置
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)
        
        # 设置设备
        self.device = torch.device(self.config['device'])
        
        # 初始化模型组件
        self.grid = None
        self.se = None
        self.mlp = None
        self.optimizer = None
        self.scheduler = None
        
        # 构建模型
        self.build_model()

        # 添加损失记录变量
        self.deform_losses = []  # 记录形变损失
        self.smooth_losses = []  # 记录平滑损失
        self.total_losses = []   # 记录总损失
    
    def build_model(self):
        """构建网络模型和优化器"""
        H, W = self.config['height'], self.config['width']
        
        # 创建坐标网格
        self.grid = create_meshgrid(H, W, self.device)
        
        # 空间嵌入层
        self.se = SpatialEmbed(
            random_seed=self.config['random_seed'],
            nB=self.config['encoding_dim'],
            beta=self.config['beta']
        ).to(self.device)

        trainable_params = []
        # 如果beta为None，SpatialEmbed的B参数需要被训练
        if self.config['beta'] is None:
            trainable_params.extend(self.se.parameters())
            # print("注意: beta=None，空间编码参数B将被训练")
            
        # MLP卷积网络
        self.mlp = MLP_Conv(in_channels=self.config['encoding_dim']*2, hidden_scale=2, out_channels=2).to(self.device)

        trainable_params.extend(self.mlp.parameters())
        
        # 优化器和学习率调度器
        self.optimizer = optim.Adam(trainable_params, lr=self.config['lr'])
        self.scheduler = StepLR(self.optimizer, step_size=self.config['attenuation_step'], gamma=self.config['attenuation_factor'])

    # 主损失函数的类型选择
    def get_deform_loss(self, warped_prev_vg, warped_last_vg):
    
        deform_loss_type = self.config['deform_loss_type']
        
        if deform_loss_type == 'l1':
            loss_fn = nn.L1Loss()
            return loss_fn(warped_prev_vg, warped_last_vg)
        
        elif deform_loss_type == 'l2':
            loss_fn = nn.MSELoss()  # PyTorch 中的 L2 损失是 MSELoss
            return loss_fn(warped_prev_vg, warped_last_vg)
        
        elif deform_loss_type == 'trick':
            eps = self.config.get('trick_eps', 0.001)
            alpha = self.config.get('trick_alpha', 0.5)
            
            def trick_loss(warped_prev_vg, warped_last_vg):
                diff = warped_prev_vg - warped_last_vg
                return torch.pow(torch.square(diff) + eps**2, alpha).mean()
    
            return trick_loss(warped_prev_vg, warped_last_vg)
    
        else:
            raise ValueError(f"不支持的形变损失类型: {deform_loss_type}")


    # 光滑损失函数类型选择
    def get_smoothness_loss(self, flow):
        smooth_loss_type = self.config['smooth_loss_type']
        
        dx = flow[:, :, :, :-1] - flow[:, :, :, 1:]
        dy = flow[:, :, :-1, :] - flow[:, :, 1:, :]
        
        if smooth_loss_type == 'l1':
            loss = dx.abs().mean() + dy.abs().mean()
        elif smooth_loss_type == 'l2':
            loss = (dx ** 2).mean() + (dy ** 2).mean()
        else:
            raise ValueError(f"Unsupported smooth loss type: {smooth_loss_type}")
        
        return loss

    # 光滑损失系数，固定值或可变值
    def get_smooth_param(self, epoch, initial_smooth_param=1, step=2000, decay_factor=0.5):
        """阶梯衰减光滑损失系数，每2000步衰减为之前的0.5"""
        decay_steps = epoch // step  # 计算当前已经经历了多少个衰减周期
        smooth_param = initial_smooth_param * (decay_factor ** decay_steps)
        return smooth_param
        
    def warp_voxel_grid(self, voxel_grid, displacement, mode='forward'):
        H, W = self.config['height'], self.config['width']
        
        if mode == 'forward':
            new_grid = self.grid - displacement
        elif mode == 'backward':
            new_grid = self.grid + displacement
        else:
            raise ValueError(f"Unsupported mode: {mode}, should be 'forward' or 'backward'")

        # 归一化到 [-1, 1]
        norm_grid = new_grid.permute(0, 2, 3, 1).clone()  # [B, H, W, 2]
        norm_grid[..., 0] = 2.0 * norm_grid[..., 0] / (W - 1) - 1.0
        norm_grid[..., 1] = 2.0 * norm_grid[..., 1] / (H - 1) - 1.0

        warped = F.grid_sample(voxel_grid, norm_grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        return warped
    
    def plot_loss_curve(self, save_path=None, log_scale=True):
       
        if not hasattr(self, 'total_losses') or len(self.total_losses) == 0:
            print("Warning: No loss data is available. Please run training first.")
            return

        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.total_losses) + 1)
        
        # 绘制曲线
        plt.plot(epochs, self.total_losses, 'b-', linewidth=2, label='total_loss')
        plt.plot(epochs, self.deform_losses, 'g--', label='deform_losses')
        plt.plot(epochs, self.smooth_losses, 'r--', label='smooth_losses')

        # 设置图表属性
        plt.title('loss_curve', fontsize=14)
        plt.xlabel('epoch', fontsize=12)
        plt.ylabel('loss', fontsize=12)
        if log_scale:
            plt.yscale('log')
        plt.legend(fontsize=12)
        plt.grid(True, which="both", ls="-", alpha=0.3)
        
        # 自动调整x轴范围
        if len(epochs) > 10:
            plt.xlim(1, len(epochs))
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"损失曲线已保存至: {save_path}")
        else:
            plt.show()
        plt.close()

    def show_information(self):
        print("=" * 60)
        print("开始训练事件光流网络")
        print(f"设备: {self.device}")
        print(f"图像尺寸: {self.config['width']}x{self.config['height']}")
        print(f"空间编码维度: {self.config['encoding_dim']}")
        print(f"光滑损失系数：{self.config['smooth_param']}")
        print(f"训练轮数: {self.config['epochs']}")
        print(f"学习率: {self.config['lr']} (衰减因子: {self.config['attenuation_factor']})")
        print("=" * 60)

    def predict_flow(self):
        """预测光流场"""
        self.mlp.eval()
        self.se.eval()
        with torch.no_grad():
            gamma = self.se(self.grid)
            flow = self.mlp(gamma).cpu().numpy()[0]
            u, v = flow[1], flow[0]
        return u, v

# =============== 辅助函数和类 ===============

def create_meshgrid(height, width, device="cuda"):
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(0, width - 1, width, device=device),
        torch.linspace(0, height - 1, height, device=device),
        indexing='xy'
    )
    grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0)  # [1, 2, H, W]
    return grid

class SpatialEmbed(nn.Module):
    def __init__(self, random_seed=None, nB=32, beta=700):
        super().__init__()
        torch.manual_seed(random_seed)
        rng_agent2 = np.random.default_rng(random_seed)
        
        if beta is None:
            B_np = (1 / 100) * rng_agent2.normal(size=(2, nB))
            B_tensor = torch.tensor(B_np.T, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.B = nn.Parameter(B_tensor)
        else:
            B_np = (1 / beta) * rng_agent2.normal(size=(2, nB))
            B_tensor = torch.tensor(B_np.T, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
            self.register_buffer('B', B_tensor)
    
    def forward(self, x):
        z = F.conv2d(x, self.B.to(x.device), bias=None)
        emb = torch.cat((torch.sin(z), torch.cos(z)), dim=1)
        return emb

class MLP_Conv(nn.Module):
    def __init__(self, in_channels=64, hidden_scale=2, out_channels=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels*hidden_scale, kernel_size=1)        
        self.conv2 = nn.Conv2d(in_channels*hidden_scale, in_channels*hidden_scale, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels*hidden_scale, in_channels//hidden_scale, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels//hidden_scale, out_channels, kernel_size=1)
        
    def forward(self, gamma):
        h1 = self.conv1(gamma)
        h1 = torch.sin(h1)
        h2 = self.conv2(h1)
        h2 = torch.sin(h2)
        h3 = self.conv3(h2)
        h3 = torch.sin(h3)
        return self.conv4(h3)

# ========================NIR-VG=================================

class NIR_VG(NIR_Base):
    def train_step(self, prev_vg, last_vg, epoch):
        """执行单次训练迭代"""
        self.mlp.train()
        self.se.train()
        self.optimizer.zero_grad()

        # 获取光滑损失系数
        if self.config['smooth_param'] is None:
            smooth_param = self.get_smooth_param(epoch)  # 使用阶梯衰减
        else:
            smooth_param = self.config['smooth_param']  # 使用提供的数值

        gamma = self.se(self.grid)
        # 预测位移场
        dis = self.mlp(gamma)  # [B, 2, H, W]
        half_dis = 0.5 * dis

        # 变形体素网格
        warped_prev_vg = self.warp_voxel_grid(prev_vg, half_dis, mode='forward')
        warped_last_vg = self.warp_voxel_grid(last_vg, half_dis, mode='backward')

        # 计算损失
        deform_loss = self.get_deform_loss(warped_prev_vg, warped_last_vg)
        loss_smooth = self.get_smoothness_loss(dis)
        loss = deform_loss + smooth_param * loss_smooth

        # 反向传播
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return deform_loss.item(), loss_smooth.item(), loss.item()
    
    def estimate_flow(self, prev_vg, last_vg):
        """
        完整训练流程
        参数: prev_vg, last_vg (Tensor): 第一、二个时间段体素网格 [1, C, H, W]
        返回: flow (np.ndarray): 最终预测的光流场 [2, H, W]
        """
        num_epochs = self.config['epochs']
        verbose = self.config.get('verbose', False)
        start_time = time.time()

        self.deform_losses = []
        self.smooth_losses = []
        self.total_losses = []

        if verbose:
            self.show_information()

        for epoch in range(1, num_epochs + 1):
            deform_loss, smooth_loss, total_loss = self.train_step(prev_vg, last_vg, epoch)

            self.deform_losses.append(deform_loss)
            self.smooth_losses.append(smooth_loss)
            self.total_losses.append(total_loss)

            if verbose and (epoch % 1000 == 0 or epoch == num_epochs):
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:05d}/{num_epochs} | LR: {lr:.6f} | "
                      f"Voxel Loss: {deform_loss:.6f} | "
                      f"Smooth Loss: {smooth_loss:.6f} | "
                      f"Total Loss: {total_loss:.6f}")

        end_time = time.time()
        if verbose:
            print("\n训练完成!")
            print(f"总训练时间: {end_time - start_time:.2f} 秒")
            
            self.plot_loss_curve()

        return self.predict_flow()

# ======================NIR-GT======================================

class NIR_GT(NIR_Base):
    """使用GT真值进行训练的NIR类"""
    
    def train_step(self, gt_flow, epoch):
        self.mlp.train()
        self.se.train()
        self.optimizer.zero_grad()

        if self.config['smooth_param'] is None:
            smooth_param = self.get_smooth_param(epoch)
        else:
            smooth_param = self.config['smooth_param']

        gamma = self.se(self.grid)
        pred_flow = self.mlp(gamma)

        deform_loss = self.get_deform_loss(pred_flow, gt_flow)
        loss_smooth = self.get_smoothness_loss(pred_flow)
        loss = deform_loss + smooth_param * loss_smooth

        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return deform_loss.item(), loss_smooth.item(), loss.item()
    
    def estimate_flow(self, gt_flow=None):
        num_epochs = self.config['epochs']
        verbose = self.config.get('verbose', False)
        start_time = time.time()

        self.deform_losses = []
        self.smooth_losses = []
        self.total_losses = []

        if verbose:
            self.show_information()

        for epoch in range(1, num_epochs + 1):
            deform_loss, smooth_loss, total_loss = self.train_step(gt_flow, epoch)

            self.deform_losses.append(deform_loss)
            self.smooth_losses.append(smooth_loss)
            self.total_losses.append(total_loss)

            if verbose and (epoch % 1000 == 0 or epoch == num_epochs):
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:05d}/{num_epochs} | LR: {lr:.6f} | "
                      f"GT Loss: {deform_loss:.6f} | "
                      f"Smooth Loss: {smooth_loss:.6f} | "
                      f"Total Loss: {total_loss:.6f}")

        end_time = time.time()
        if verbose:
            print("\n训练完成!")
            print(f"总训练时间: {end_time - start_time:.2f} 秒")

        if verbose:
            self.plot_loss_curve()

        return self.predict_flow()

# =======================================================================================
#                        OpenCV's optical flow module : Farneback
# =======================================================================================
import concurrent.futures

class Farneback:
    def __init__(self, config=None):
        self.default_config = {
            'pyr_scale': 0.5,
            'levels': 4,
            'winsize': 25,
            'iterations': 5,
            'poly_n': 7,
            'poly_sigma': 1.5,
            'clahe_clip_limit': 2.0,
        }
        
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clahe = cv2.createCLAHE(clipLimit=self.config['clahe_clip_limit'], tileGridSize=(8, 8))

    def enhance_voxel_channel(self, channel):
        norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = self.clahe.apply(norm)
        smooth = cv2.GaussianBlur(clahe, (5, 5), 1.5)
        return smooth

    def calc_flow(self, img1, img2):
        try:
            return cv2.calcOpticalFlowFarneback(
                img1, img2, None,
                pyr_scale=self.config['pyr_scale'],
                levels=self.config['levels'],
                winsize=self.config['winsize'],
                iterations=self.config['iterations'],
                poly_n=self.config['poly_n'],
                poly_sigma=self.config['poly_sigma'],
                flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN
            )
        except:
            return np.zeros((*img1.shape, 2), dtype=np.float32)

    def compute_flow_for_channel(self, v1, v2, flow_u, flow_v, c):
        """
        计算单通道的光流
        """
        i1 = self.enhance_voxel_channel(v1[c])
        i2 = self.enhance_voxel_channel(v2[c])
        flow = self.calc_flow(i1, i2)
        flow_u[c], flow_v[c] = flow[..., 0], flow[..., 1]

    def estimate_flow_single_parallel(self, voxel1, voxel2):
        """
        估计光流：单向，从 voxel1 到 voxel2，使用并行处理

        输入:
            voxel1, voxel2: torch.Tensor, shape = (1, C, H, W)

        输出:
            u, v: np.ndarray, shape = (H, W)
        """
        _, C, h, w = voxel1.shape  # 使用h（行）和w（列）
        v1, v2 = voxel1[0].cpu().numpy(), voxel2[0].cpu().numpy()

        flow_u = np.zeros((C, h, w), dtype=np.float32)
        flow_v = np.zeros((C, h, w), dtype=np.float32)

        # 使用并行处理每个通道的光流计算
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for c in range(C):
                futures.append(executor.submit(self.compute_flow_for_channel, v1, v2, flow_u, flow_v, c))
            
            # 等待所有线程完成
            for future in futures:
                future.result()

        # 计算最终的平均光流
        return np.mean(flow_u, axis=0), np.mean(flow_v, axis=0)

    def estimate_flow(self, voxel1, voxel2):
        """
        双向光流估计：forward - backward 平均

        输入:
            voxel1, voxel2: torch.Tensor, shape = (1, C, H, W)

        输出:
            u, v: np.ndarray, shape = (H, W)
        """
        u_fwd, v_fwd = self.estimate_flow_single_parallel(voxel1, voxel2)
        u_bwd, v_bwd = self.estimate_flow_single_parallel(voxel2, voxel1)

        u = 0.5 * (u_fwd - u_bwd)
        v = 0.5 * (v_fwd - v_bwd)
        return v, u

# =======================================================================================
# ILK（Iterative Lucas-Kanade）
# =======================================================================================
from skimage.registration import optical_flow_ilk
from concurrent.futures import ThreadPoolExecutor

class Iter_LK:
    # 光流计算器：基于 ILK 算法计算连续层之间的光流

    def __init__(self, config=None):
        # 默认配置
        self.default_config = {
            'resize_factor': 2,       # 下采样比例
            'max_workers': 4,         # 最大线程数
            'device': "cpu",  # 设备设置
            'radius': 16,             # ILK 半径
            'num_warp': 1,            # ILK warp 次数
            'ksize': (7, 7),          # 模糊核大小
        }
        
        # 通过传入的 config 更新默认配置
        self.config = self.default_config.copy()
        if config is not None:
            self.config.update(config)

        # 从配置中提取参数
        self.resize_factor = self.config['resize_factor']
        self.max_workers = self.config['max_workers']
        self.device = self.config['device']
        self.radius = self.config['radius']
        self.num_warp = self.config['num_warp']
        self.ksize = self.config['ksize']

        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def estimate_flow(self, voxel_grid1, voxel_grid2):
        """计算两个体素网格输入的光流，并对相同通道的光流进行平均。"""
        num_bins = voxel_grid1.shape[1]
        if num_bins < 2:
            return None

        flow_tensors = []
        futures = []

        # 对每对相同的通道计算光流
        for j in range(num_bins - 1):
            img1 = voxel_grid1[0, j].detach().cpu().numpy()  # 从第一个输入获取图像
            img2 = voxel_grid2[0, j].detach().cpu().numpy()  # 从第二个输入获取图像
            futures.append(self.executor.submit(self._compute_ilk_flow_tensor, img1, img2))

        # 处理每个计算结果
        for future in futures:
            flow = future.result()
            if flow is not None:
                flow_tensors.append(flow)

        # 对所有计算的光流进行堆叠并取平均
        if flow_tensors:
            flow = torch.stack(flow_tensors, dim=0).mean(dim=0).cpu().numpy()  # 在第0维上求平均
            return flow[0], flow[1]
            
        return None

    def _compute_ilk_flow_tensor(self, img1, img2):
        """使用 ILK 计算光流 (2, H, W)"""
        rows, cols = img1.shape
        img1 = cv2.GaussianBlur(img1, self.ksize, 0)
        img2 = cv2.GaussianBlur(img2, self.ksize, 0)

        # 调整图像大小
        new_size = (int(cols / self.resize_factor), int(rows / self.resize_factor))
        img1 = cv2.resize(img1, new_size).astype(np.float32)
        img2 = cv2.resize(img2, new_size).astype(np.float32)

        u, v = optical_flow_ilk(
            img1, img2,
            radius=self.radius,
            num_warp=self.num_warp,
            gaussian=False,
            prefilter=True
        )

        # 恢复原始大小
        u = cv2.resize(u, (cols, rows))
        v = cv2.resize(v, (cols, rows))

        return torch.from_numpy(np.stack([u, v], axis=0)).float()

