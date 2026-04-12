import math
import torch
from addict import Dict

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def config_image():
    cfg = Dict()
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    cfg.density = 0.05
    cfg.width = 256
    cfg.height = 256
    cfg.d = 1.0
    cfg.d_std = 0.2
    cfg.l = 200
    cfg.l_std = 10
    return cfg

def config_sequence():
    cfg = config_image()
    cfg.time_length = 15   # in ms;
    return cfg

def config_events():
    cfg = config_sequence()
    cfg.threshold = 0.25 # the constant fire threshold
    return cfg


def render_particles_image(x, y, d, l, W=256, H=256, device=device):
    # 删除不在范围内的数据
    mask = (x >= -10) & (x < H + 10) & (y >= -10) & (y < W + 10)
    x,y = x[mask], y[mask]
    d,l = d[mask], l[mask]

    # 输入数据
    B = x.shape[0]

    # patch 相关
    sigma= d/4.0
    M = int(torch.ceil(6 * torch.max(sigma) + 3).item()) # patch_size
    # pad = M//2 + 1
    pad = M//2
    
    # 构建patch网格
    ax = torch.arange(M, device=device) - pad  # [-pad,...,pad]
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')  # (M, M)

    # 偏移坐标 (B, M, M)
    lx = xx.view(1,M,M) - torch.frac(x).view(B,1,1)
    ly = yy.view(1,M,M) - torch.frac(y).view(B,1,1)

    # 积分近似
    erf_b = math.sqrt(2)*sigma
    ex = torch.erf((lx + 0.5) / erf_b.view(B,1,1)) - torch.erf((lx - 0.5) / erf_b.view(B,1,1))
    ey = torch.erf((ly + 0.5) / erf_b.view(B,1,1)) - torch.erf((ly - 0.5) / erf_b.view(B,1,1))
    area = ex * ey*sigma.view(B,1,1)**2*math.pi/2  # (B, M, M)
    # print(ex.dtype)

    # 归一化并给定亮度
    patch =  l.view(B,1,1) *area /torch.amax(area, dim=(1,2),keepdim=True) # (B, M, M)

    # 将 patch 写入图像：全图加 padding
    padded_W, padded_H = W + 2 * pad, H + 2 * pad
    image = torch.zeros((padded_H, padded_W), device=device, dtype=torch.float32)

    # 每个patch左上角位置：考虑padding偏移
    x0 = torch.floor(x).to(torch.long)+ pad
    y0 = torch.floor(y).to(torch.long)+ pad

    # 构建patch索引
    ix = x0.view(B, 1, 1) + xx.view(1,M,M)  # (B, M, M)
    iy = y0.view(B, 1, 1) + yy.view(1,M,M)  # (B, M, M)

    mask = (ix < 0) | (ix >= padded_H) | (iy < 0) | (iy >= padded_W)
    mask_valid = ~mask.view(-1)

    # 将 (ix, iy) 转为扁平索引
    linear_idx = ix * padded_W + iy  # (B, M, M)
    linear_idx = linear_idx.view(-1)  # (B*M*M,)
    patch_flat = patch.view(-1)      # (B*M*M,)

    # scatter_add 到图像（flatten方式）
    image_flat = image.view(-1)
    image_flat.scatter_add_(0, linear_idx[mask_valid], patch_flat[mask_valid])

    # 最后裁剪回原图大小
    final_image = image[pad:pad + H, pad:pad + W]  # (W, H)
    return final_image


class ParticleImageSimulator:
    def __init__(self, flow_func=None, cfg=config_image()):
        self._c = cfg
        self.flow_func = flow_func

    def forward(self,cfg=None):
        if cfg is not None:
            self._c = cfg
        pts = self._random_particles()
        pts_list, ts = self.update_particles(pts)
        imgs = [render_particles_image(*p,self._c.width,self._c.height) for p in pts_list]
        return imgs, ts

    def _random_particles(self):
        h, w = self._c.height, self._c.width
        H, W = 3*h, 3*w
        
        num = int(H*W*self._c.density + 0.5)
        x = torch.rand(num, device=self._c.device) * (H - 1)-h  
        y = torch.rand(num, device=self._c.device) * (W - 1)-w     
        d = torch.abs(torch.randn(num, device=self._c.device)*self._c.d_std/3.0 + self._c.d)
        l = torch.randn(num, device=self._c.device)*self._c.l_std/3.0 + self._c.l
        return (x,y,d,l) # The initial pts

    def update_particles(self,pts):
        x,y,d,l = pts
        pts_list = [pts]
        for i in range(1):
            u,v = self.flow_func(x,y)
            x,y = x+u, y+v
            pts_list.append((x,y,d,l))
        ts = [0,1] # at two time 
        return pts_list, ts 


class ParticleSequenceSimulator(ParticleImageSimulator):
    def __init__(self, flow_func=None, cfg=config_sequence()):
        super().__init__(flow_func=flow_func, cfg=cfg)
        
    def update_particles(self,pts):
        x,y,d,l = pts
        u,v = self.flow_func(x,y)
        max_v = torch.max(u.abs().max(), v.abs().max())
        delta_t = 0.05/max_v
        # print(delta_t, self._c.time_length)
        ts = torch.arange(0, self._c.time_length+ 1e-8, delta_t, device=self._c.device)  # 加一个小数防止丢最后一个点

        K = 1
        pts_list = [pts]
        for t in ts[1:]:
            # x, y = x.clone(), y.clone()
            for k in range(K):
                u,v = self.flow_func(x,y)
                x,y = x+u*delta_t/K, y+v*delta_t/K
            pts_list.append((x,y,d,l))
        return pts_list, ts 
        

class Sequence2events():
    def __init__(self,flow_func,cfg=config_events()):
        self.pss = ParticleSequenceSimulator(flow_func=flow_func,cfg=cfg)
        self._c = cfg
        self._flow_func = flow_func

    def load(self): # alias
        evts = self.forward()
        x_vals = torch.linspace(0, self._c.height-1, steps=self._c.height, device=self._c.device)
        y_vals = torch.linspace(0, self._c.width-1, steps=self._c.width, device=self._c.device)
        x, y = torch.meshgrid(x_vals, y_vals, indexing='ij')
        u, v = self._flow_func(x, y)
        return evts, (u,v)

    def forward(self):
        imgs_list, ts = self.pss.forward() # 生成一系列的图像，给出ts标签
        return self.imgs2evts(imgs_list, ts)


    def imgs2evts(self, imgs_list, ts):
        imgs = torch.stack(imgs_list,dim=0)
        # print(imgs.shape, len(ts))

        array = torch.log(imgs+1)

        prev_vals = array[:-1]     # [T-1, H, W]
        next_vals = array[1:]      # [T-1, H, W]
        diffs = next_vals - prev_vals
    
        max_val = array.max()
        c_list = torch.arange(0, 1.0, self._c.threshold, device=self._c.device) * max_val
        
        all_events = []
        for c in c_list:
            # mask of contrast crossing
            mask_up = (prev_vals <= c) & (c <= next_vals)
            mask_down = (next_vals <= c) & (c <= prev_vals)
            mask = (mask_up | mask_down) & (diffs != 0)
    
            if not mask.any():
                continue
    
            indices = mask.nonzero(as_tuple=False)  # shape [N, 3] → (t_idx, x, y)
            i_indices = indices[:, 0]
            x_indices = indices[:, 1]
            y_indices = indices[:, 2]
    
            a = prev_vals[i_indices, x_indices, y_indices]
            b = next_vals[i_indices, x_indices, y_indices]
    
            proportions = (c - a) / (b - a + 1e-12)
            t0 = ts[i_indices]
            t1 = ts[i_indices + 1]
            t_values = t0 + proportions * (t1 - t0)
    
            polars = ((a <= c) & (c <= b)).to(torch.int8)  # ON=1, OFF=0
    
            events = torch.stack([x_indices.float(),
                                  y_indices.float(),
                                  t_values,
                                  polars.float()], dim=1)  # shape [N, 4]
    
            all_events.append(events)
    
        if all_events:
            all_events = torch.cat(all_events, dim=0)  # shape [N, 4]
            
            # 排序关键：第3列是时间戳 t
            t_values = all_events[:, 2]  # shape [N]
            sorted_indices = torch.argsort(t_values)
            all_events = all_events[sorted_indices]  # 按 t 升序排序后的事件
            return all_events

        else:
            return torch.zeros((0, 4), dtype=torch.float32, device=array.device)

