import numpy as np
import cv2
from skimage.registration import optical_flow_ilk

import torch.nn.functional as F
import torch
from INR.frames import events_to_voxel_grid, events_to_est
from INR.inr import MLP_Conv, train_func
from gaussian_flow.config import GaussianFlowConfig
from gaussian_flow.training import run_gaussian_flow

# ====================== PyTorch ============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def affineGS(evts, H=256, W=256):
    cfg = GaussianFlowConfig(verbose=False,H=H, W=W)
    events = evts[(evts[:, 2] > 0) & (evts[:, 2] < 3)]
    result = run_gaussian_flow(events=events, cfg=cfg, flowgt=None)
    u, v = result.flow[0], result.flow[1]
    div = result.div
    vor = result.vorticity
    return u, v, div, vor

def constGS(evts, H=256, W=256):
    cfg = GaussianFlowConfig(verbose=False, H=H, W=W, enable_jacobian=False)
    events = evts[(evts[:, 2] > 0) & (evts[:, 2] < 3)]
    result = run_gaussian_flow(events=events, cfg=cfg, flowgt=None)
    u, v = result.flow[0], result.flow[1]
    div = result.div
    vor = result.vorticity
    return u, v, div, vor

def inr_vg(evts, H=256, W=256):
    vg, dts = events_to_voxel_grid(evts, 21, H=H, W=W)
    # vg = (vg*255/(torch.max(vg)+1e-8))
    dt = (dts/20).item()
        
    blur = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    vg = vg.unsqueeze(1)     # B×1×W×H
    vg = blur(vg)
    vg = vg.squeeze(1)       # B×W×H

    x,y = default_grids(height=H, width=W)
    xs = torch.concat([x, y], dim=1).to(device)  # (1, 2, H, W)
    
    inr_model = MLP_Conv(ne=64, beta=100).to(device)
    trained_model = train_func(model=inr_model, xs=xs, vg=vg, dt=1.0, 
                             num_epochs=1000, learning_rate=0.001, lambda_smooth=10.0, verbose=False)
    u, v = inr_model(xs,dt=dt)
    u, v = u.squeeze().detach().cpu().numpy(), v.squeeze().detach().cpu().numpy()
    vx, vy, dvx_dx, dvx_dy, dvy_dx, dvy_dy = trained_model.forward_manual_grad(xs,dt=dt)
    div = dvx_dy + dvy_dx
    vor = dvx_dx - dvy_dy
    return u, v, div.squeeze(0).squeeze(0).cpu(), -vor.squeeze(0).squeeze(0).cpu()

def inr_est(evts, H=256, W=256):
    vg, dts = events_to_est(evts, 21, H=H, W=W)
    dt = (dts/20).item()

    blur = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    vg = vg.unsqueeze(1)     # B×1×W×H
    vg = blur(vg)
    vg = vg.squeeze(1)       # B×W×H

    x,y = default_grids(height=H, width=W)
    xs = torch.concat([x, y], dim=1).to(device)  # (1, 2, H, W)
    
    inr_model = MLP_Conv(ne=64, beta=100).to(device)
    trained_model = train_func(model=inr_model, xs=xs, vg=vg, dt=1.0, 
                             num_epochs=1000, learning_rate=0.001, lambda_smooth=10.0, verbose=False)
    u, v = inr_model(xs, dt=dt)
    u, v = u.squeeze().detach().cpu().numpy(), v.squeeze().detach().cpu().numpy()
    vx, vy, dvx_dx, dvx_dy, dvy_dx, dvy_dy = trained_model.forward_manual_grad(xs,dt=dt)
    div = dvx_dy + dvy_dx
    vor = dvx_dx - dvy_dy
    
    return u, v, div.squeeze(0).squeeze(0).cpu(), -vor.squeeze(0).squeeze(0).cpu()


CLASSICAL_NUM_BINS = 21
CLASSICAL_PAIR_OFFSET = 5
FARNEBACK_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _normalize_voxel_grid(vg):
    return (vg * 255.0 / (torch.max(vg) + 1e-8)).detach().cpu().numpy().clip(0, 255).astype(np.uint8)


def _enhance_farneback_frame(frame):
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    frame = FARNEBACK_CLAHE.apply(frame)
    return cv2.GaussianBlur(frame, (5, 5), 1.5)


def _resize_flow_frames(vg, resize_factor):
    resize_factor = float(resize_factor)
    if resize_factor <= 1.0:
        return vg, 1.0
    new_w = max(2, int(round(vg.shape[2] / resize_factor)))
    new_h = max(2, int(round(vg.shape[1] / resize_factor)))
    return np.stack([cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA) for frame in vg]), resize_factor


def _restore_flow_resolution(flow, H, W, resize_factor):
    if resize_factor <= 1.0:
        return flow
    flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
    return flow * resize_factor


def of_vg(evts, H=256, W=256):
    vg, dts = events_to_voxel_grid(evts, CLASSICAL_NUM_BINS, H=H, W=W)
    dt = (dts/(CLASSICAL_NUM_BINS - 1)).item()
    vg = _normalize_voxel_grid(vg)
    
    flows = []
    for t in range(vg.shape[0] - CLASSICAL_PAIR_OFFSET):
        img1 = _enhance_farneback_frame(vg[t])
        img2 = _enhance_farneback_frame(vg[t + CLASSICAL_PAIR_OFFSET])
        f = cv2.calcOpticalFlowFarneback(img1, img2, None,
            0.5, 4, 25, 5, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flows.append(f)
    
    flow_mean = np.mean(flows, axis=0)
    u = flow_mean[..., 1]/(CLASSICAL_PAIR_OFFSET*dt)
    v = flow_mean[..., 0]/(CLASSICAL_PAIR_OFFSET*dt)
    div, vor = compute_flow_quantities(u, v)
    return u, v, div, vor

def of_est(evts, H=256, W=256):
    vg, dts = events_to_est(evts, CLASSICAL_NUM_BINS, H=H, W=W)
    dt = (dts/(CLASSICAL_NUM_BINS - 1)).item()
    vg = _normalize_voxel_grid(vg)
    
    flows = []
    for t in range(vg.shape[0] - CLASSICAL_PAIR_OFFSET):
        img1 = _enhance_farneback_frame(vg[t])
        img2 = _enhance_farneback_frame(vg[t + CLASSICAL_PAIR_OFFSET])
        f = cv2.calcOpticalFlowFarneback(img1, img2, None,
            0.5, 4, 25, 5, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flows.append(f)
    
    flow_mean = np.mean(flows, axis=0)
    u = flow_mean[..., 1]/(CLASSICAL_PAIR_OFFSET*dt)
    v = flow_mean[..., 0]/(CLASSICAL_PAIR_OFFSET*dt)
    div, vor = compute_flow_quantities(u, v)
    return u, v, div, vor


def ilk_vg(evts, radius=7, num_warp=5, H=256, W=256):
    vg, dts = events_to_voxel_grid(evts, CLASSICAL_NUM_BINS, H=H, W=W)
    dt = (dts/(CLASSICAL_NUM_BINS - 1)).item()
    vg = _normalize_voxel_grid(vg)

    flows_u, flows_v = [], []

    for t in range(vg.shape[0] - CLASSICAL_PAIR_OFFSET):
        img1 = vg[t].astype(np.float32)
        img2 = vg[t + CLASSICAL_PAIR_OFFSET].astype(np.float32)

        u, v = optical_flow_ilk(
            img1, img2,
            radius=radius,
            num_warp=num_warp,
            gaussian=False,
            prefilter=True
        )

        flows_u.append(u)
        flows_v.append(v)

    # temporal averaging
    u_mean = np.mean(flows_u, axis=0)/(CLASSICAL_PAIR_OFFSET*dt)
    v_mean = np.mean(flows_v, axis=0)/(CLASSICAL_PAIR_OFFSET*dt)
    div, vor = compute_flow_quantities(u_mean, v_mean)
    
    return u_mean, v_mean, div, vor

def ilk_est(evts, radius=7, num_warp=5, H=256, W=256):
    vg, dts = events_to_est(evts, CLASSICAL_NUM_BINS, H=H, W=W)
    dt = (dts/(CLASSICAL_NUM_BINS - 1)).item()
    vg = _normalize_voxel_grid(vg)

    flows_u, flows_v = [], []

    for t in range(vg.shape[0] - CLASSICAL_PAIR_OFFSET):
        img1 = vg[t].astype(np.float32)
        img2 = vg[t + CLASSICAL_PAIR_OFFSET].astype(np.float32)

        u, v = optical_flow_ilk(
            img1, img2,
            radius=radius,
            num_warp=num_warp,
            gaussian=False,
            prefilter=True
        )

        flows_u.append(u)
        flows_v.append(v)

    # temporal averaging
    u_mean = np.mean(flows_u, axis=0)/(CLASSICAL_PAIR_OFFSET*dt)
    v_mean = np.mean(flows_v, axis=0)/(CLASSICAL_PAIR_OFFSET*dt)
    div, vor = compute_flow_quantities(u_mean, v_mean)
    
    return u_mean, v_mean, div, vor


def default_grids(height=256, width=512):
    # basic coordinate
    y_vals = torch.linspace(0, height-1, steps=height)
    x_vals = torch.linspace(0, width-1, steps=width)
    y, x = torch.meshgrid(y_vals, x_vals, indexing='ij')
    x, y = x[None, None].to(device), y[None, None].to(device)
    return x,y 


def compute_flow_quantities(v_h, v_w):
    """
    计算流场的散度和涡量
    v_h: 竖直方向速度分量 [H, W] 或 [1, H, W] 或 [1, 1, H, W]
    v_w: 水平方向速度分量 [H, W] 或 [1, H, W] 或 [1, 1, H, W]
    """
    # 转为 torch 张量（如果还不是）
    if not isinstance(v_h, torch.Tensor):
        v_h = torch.tensor(v_h, dtype=torch.float32)
        v_w = torch.tensor(v_w, dtype=torch.float32)
    
    # 统一形状为 (1, 1, H, W)
    if v_h.dim() == 2:          # (H, W)
        v_h = v_h.unsqueeze(0).unsqueeze(0)
        v_w = v_w.unsqueeze(0).unsqueeze(0)
    elif v_h.dim() == 3:        # (batch, H, W) 或 (channel, H, W)
        v_h = v_h.unsqueeze(1)  # 添加通道维度
        v_w = v_w.unsqueeze(1)
    elif v_h.dim() == 4:        # 已经是 (batch, channel, H, W)
        pass
    else:
        raise ValueError(f"Unsupported input dimension: {v_h.dim()}")
    
    # 填充，形状为 (1,1,H,W)，支持 replicate
    v_h_pad = F.pad(v_h, (1, 1, 1, 1), mode='replicate')
    v_w_pad = F.pad(v_w, (1, 1, 1, 1), mode='replicate')
    
    # 计算一阶导数（中心差分）
    # 索引说明: 
    #   - dim 2: 高度方向 (h)
    #   - dim 3: 宽度方向 (w)
    dv_h_dh = (v_h_pad[:, :, 2:, 1:-1] - v_h_pad[:, :, :-2, 1:-1]) / 2      # ∂v_h/∂h
    dv_h_dw = (v_h_pad[:, :, 1:-1, 2:] - v_h_pad[:, :, 1:-1, :-2]) / 2      # ∂v_h/∂w
    dv_w_dh = (v_w_pad[:, :, 2:, 1:-1] - v_w_pad[:, :, :-2, 1:-1]) / 2      # ∂v_w/∂h
    dv_w_dw = (v_w_pad[:, :, 1:-1, 2:] - v_w_pad[:, :, 1:-1, :-2]) / 2      # ∂v_w/∂w
    
    # 散度: div = ∂v_h/∂h + ∂v_w/∂w
    div = dv_h_dh + dv_w_dw
    
    # 涡量: vorticity = ∂v_w/∂h - ∂v_h/∂w
    vorticity = dv_w_dh - dv_h_dw
    
    # 去掉多余维度，返回原始形状
    return div.squeeze(), vorticity.squeeze()
