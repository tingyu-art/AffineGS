import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class SharedConvPair(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_ch, in_ch, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(
            self.weight,
            mode="fan_in",
            nonlinearity="relu")

        self.bias = nn.Parameter(
            torch.zeros(out_ch))

    def conv_with_bias(self, x):
        return F.conv2d(x, self.weight, self.bias)

    def conv_no_bias(self, x):
        return F.conv2d(x, self.weight, None)


class GeLUFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.activate = nn.GELU()

    def forward(self, x):
        return self.activate(x)

    def forward_grad(self, x):
        pdf = torch.exp(-0.5 * x**2) / torch.sqrt(torch.tensor(2*torch.pi))
        grads = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0)))) + x * pdf
        return grads


class MLP_Conv(nn.Module):
    def __init__(self, ne=64, beta=10.0):
        super().__init__()
        # Embedding layer: input 2 channels -> ne//2 channels
        self.emb = SharedConvPair(2, ne // 2, kernel_size=1)
        with torch.no_grad():
            self.emb.weight.copy_(torch.randn_like(self.emb.weight) / beta)

        # Main MLP network layers
        self.conv1 = SharedConvPair(ne, ne * 2, kernel_size=1)
        self.conv2 = SharedConvPair(ne * 2, ne * 2, kernel_size=1)
        self.conv3 = SharedConvPair(ne * 2, ne // 2, kernel_size=1)
        self.conv4 = SharedConvPair(ne // 2, 2, kernel_size=1)
        self.activate = GeLUFunction()

    def forward(self, x, dt=1.0):
        emb = self.emb.conv_with_bias(x)  # shape: B, ne//2, H, W
        # Positional embedding: concatenate sin and cos
        z = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # B, ne, H, W

        # Forward through MLP Conv layers
        h1 = self.activate(self.conv1.conv_with_bias(z))
        h2 = self.activate(self.conv2.conv_with_bias(h1))
        h3 = self.activate(self.conv3.conv_with_bias(h2))
        v = self.conv4.conv_with_bias(h3)/dt
        vx,vy = torch.split(v, 1, dim=1)
        return vx, vy

    def forward_manual_grad(self, x, dt=1.0):
        """
        Forward pass with manual gradient propagation.
        Returns:
            vx, vy: velocity components (B,1,H,W)
            dvx_dx, dvx_dy: gradients of vx w.r.t x and y (B,1,H,W)
            dvy_dx, dvy_dy: gradients of vy w.r.t x and y (B,1,H,W)
        """
        # --- 构造单位方向微分 ---
        dx1 = torch.zeros_like(x)
        dx1[:, 0:1, :, :] = 1.0  # 对 x 方向
        dx2 = torch.zeros_like(x)
        dx2[:, 1:2, :, :] = 1.0  # 对 y 方向
    
        # --- Embedding ---
        emb = self.emb.conv_with_bias(x)          # B, ne//2, H, W
        emb_d1 = self.emb.conv_no_bias(dx1)
        emb_d2 = self.emb.conv_no_bias(dx2)
    
        # --- Positional embedding ---
        z = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        z_d1 = torch.cat([torch.cos(emb) * emb_d1, -torch.sin(emb) * emb_d1], dim=1)
        z_d2 = torch.cat([torch.cos(emb) * emb_d2, -torch.sin(emb) * emb_d2], dim=1)
    
        # --- 第一层卷积 ---
        h1 = self.conv1.conv_with_bias(z)
        h1_d1 = self.conv1.conv_no_bias(z_d1)
        h1_d2 = self.conv1.conv_no_bias(z_d2)
        h1_act = self.activate(h1)
        grad_h1 = self.activate.forward_grad(h1)
        h1_d1 = h1_d1 * grad_h1
        h1_d2 = h1_d2 * grad_h1
    
        # --- 第二层卷积 ---
        h2 = self.conv2.conv_with_bias(h1_act)
        h2_d1 = self.conv2.conv_no_bias(h1_d1)
        h2_d2 = self.conv2.conv_no_bias(h1_d2)
        h2_act = self.activate(h2)
        grad_h2 = self.activate.forward_grad(h2)
        h2_d1 = h2_d1 * grad_h2
        h2_d2 = h2_d2 * grad_h2
    
        # --- 第三层卷积 ---
        h3 = self.conv3.conv_with_bias(h2_act)
        h3_d1 = self.conv3.conv_no_bias(h2_d1)
        h3_d2 = self.conv3.conv_no_bias(h2_d2)
        h3_act = self.activate(h3)
        grad_h3 = self.activate.forward_grad(h3)
        h3_d1 = h3_d1 * grad_h3
        h3_d2 = h3_d2 * grad_h3
    
        # --- 输出层卷积 ---
        v = self.conv4.conv_with_bias(h3_act) / dt          # (B,2,H,W): vx, vy
        grad_x = self.conv4.conv_no_bias(h3_d1) / dt        # (B,2,H,W): ∂v/∂x
        grad_y = self.conv4.conv_no_bias(h3_d2) / dt        # (B,2,H,W): ∂v/∂y
    
        # 拆分速度分量
        vx, vy = torch.split(v, 1, dim=1)                   # each (B,1,H,W)
    
        # 拆分梯度分量
        dvx_dx, dvy_dx = torch.split(grad_x, 1, dim=1)      # ∂vx/∂x, ∂vy/∂x
        dvx_dy, dvy_dy = torch.split(grad_y, 1, dim=1)      # ∂vx/∂y, ∂vy/∂y
    
        return vx, vy, dvx_dx, dvx_dy, dvy_dx, dvy_dy

        
def warping(voxel_grid, x, y, u, v, width=512, height=256):
    x1,y1 = x-u,y-v
    x_norm = 2.0 * x1 / (width-1) - 1.0
    y_norm = 2.0 * y1 / (height-1) - 1.0
    norm_grid = torch.concat((x_norm, y_norm), dim=1).permute(0,2,3,1)
    warped = F.grid_sample(voxel_grid, norm_grid, mode='bicubic', padding_mode='zeros', align_corners=True) # 'bicubic', 'bilinear'
    return warped


def train_func(model, xs, vg, dt=1.0, num_epochs=1000, learning_rate=0.001, 
             lambda_smooth=1.0, verbose=True):
    """
    训练 INR 模型以估计速度场
    
    参数:
        vg: 输入的速度场数据，形状为 (T, H, W)
        model: INR 模型 (MLP_Conv)
        num_epochs: 训练轮数
        learning_rate: 学习率
        lambda_smooth: 平滑损失权重
        verbose: 是否打印训练信息
    """
    # 设置模型为训练模式
    model.train()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=num_epochs//10, T_mult=9)

    # 定义损失函数
    smooth_l1_loss = nn.SmoothL1Loss(beta=0.01)
    mse_loss = nn.MSELoss()
    
    # 获取输入数据的形状
    T, H, W = vg.shape
    # print(f"Shape:{T},{H},{W}")
    
    # 创建坐标网格
    x, y = torch.split(xs,1,dim=1)

    # 创建随机梯度坐标网格
    x_g = torch.rand(1, 1, 32, 32, device=xs.device) * (W - 1)
    y_g = torch.rand(1, 1, 32, 32, device=xs.device) * (H - 1)
    xs_g = torch.cat([x_g, y_g], dim=1)   # (1, 2, H, W)

    
    # 准备速度场数据
    vg_tensor = vg
    
    # 训练循环
    dts = np.linspace(1.0,dt,num_epochs)
    for epoch in range(num_epochs):
        # 全图训练模式（基于参考代码）
        optimizer.zero_grad()
        
        # forward
        u, v = model(xs,dt=dts[epoch])
        _, _, du_dx, _, _, dv_dx = model.forward_manual_grad(xs_g, dt=dts[epoch])

        # Training the data
        loss_data = 0
        for b in [1,5,6]:
            # 准备速度场数据
            vg1 = vg_tensor[:-b, :, :].unsqueeze(0)  # (1, T-1, H, W)
            vg2 = vg_tensor[b:, :, :].unsqueeze(0)    # (1, T-1, H, W)
            
            # 计算数据损失
            w1 = warping(vg1, x, y, dts[epoch]*b*v/2, dts[epoch]*b*u/2, width=W, height=H)
            w2 = warping(vg2, x, y, -dts[epoch]*b*v/2, -dts[epoch]*b*u/2, width=W, height=H)
            loss_data = loss_data+ smooth_l1_loss(w1 - w2, torch.zeros_like(w1))        
        
        loss_smooth = mse_loss(du_dx, torch.zeros_like(du_dx)) + \
                     mse_loss(dv_dx, torch.zeros_like(dv_dx))
        
        # # 总损失
        # loss_smooth = torch.zeros_like(loss_data)
        total_loss = loss_data + lambda_smooth * loss_smooth
        # total_loss = loss_data
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        scheduler.step(epoch)  # 按 step 更新学习率

        # 打印训练信息
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Data Loss: {loss_data.item():.4e}, "
                  f"Smooth Loss: {loss_smooth.item():.4e}, "
                  f"Total Loss: {total_loss.item():.4e}")
    
    # 设置模型为评估模式
    model.eval()
    return model