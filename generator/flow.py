import numpy as np
import torch
import torch.nn.functional as F 


class FlowField:
    def __init__(self, data_path="./data/backstep_Re1000_00004.npz",device=None, num_iter=500, tol=1e-4):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.img1, self.img2, self.u, self.v = self._read_data(data_path)
        self.h, self.w = self.u.shape

        self.u_ext, self.v_ext = self._extend_flow_with_coords(
            self.u, self.v, num_iter=num_iter, tol=tol)

    def _read_data(self, d_path):
        data = np.load(d_path)
        img1, img2, u, v = data['img1'], data['img2'], data['u'], data['v']
        img1, img2, u, v = torch.tensor(img1), torch.tensor(img2), torch.tensor(u), torch.tensor(v)
        img1, img2, u, v = img1.to(self.device), img2.to(self.device), u.to(self.device), v.to(self.device)
        # u,v = v.t(),u.t()
        # u,v = v,u
        return img1, img2, u, v

    def _extend_flow_with_coords(self, u, v, x=None, y=None, num_iter=600, tol=1e-4):    
        def stream_recon(u,v,k=10):
            if k<0:
                return 0
            # 获得流函数
            u_mean = torch.zeros_like(u)
            v_mean = torch.zeros_like(u)
            u_mean[1:] = (u[:-1] + u[1:])
            u_psi = torch.cumsum(u_mean,dim=0)
            v_mean[:,1:] = (v[:,:-1] + v[:,1:])
            v_psi = torch.cumsum(-v_mean,dim=1)
            psi2 = (u_psi+v_psi)/4.0
        
            # 计算流函数的速度 u= phi_x, v= phi_y
            psi = F.pad(psi2.view(1,1,self.h,self.w),(1,1,1,1),mode="replicate").squeeze()
            u_ =  (psi[2:,1:-1]-psi[:-2,1:-1])/2
            v_ = -(psi[1:-1,2:]-psi[1:-1,:-2])/2
        
            u_res, v_res = u-u_, v-v_
            return psi2 + stream_recon(u_res,v_res, k-1)
            
        psi = stream_recon(u,v)
        
        # Laplace 延拓3倍
        H, W = 3 * self.h, 3 * self.w
        psi_ext = torch.full((H, W), np.nan, device=self.device)
        psi_ext[self.h+1:2*self.h-1, self.w+1:2*self.w-1] = psi[1:-1,1:-1]
        fixed_mask = ~torch.isnan(psi_ext)
        
        for _ in range(num_iter):
            psi_ext[0,:], psi_ext[-1,:], psi_ext[:,0], psi_ext[:,-1] = 0, 0, 0, 0
            up = torch.roll(psi_ext, 1, dims=0)
            down = torch.roll(psi_ext, -1, dims=0)
            left = torch.roll(psi_ext, 1, dims=1)
            right = torch.roll(psi_ext, -1, dims=1)
            neighbor_mean = torch.nanmean(torch.stack([psi_ext, up, down, left, right]), dim=0)
            psi_ext[~fixed_mask] = neighbor_mean[~fixed_mask]
                    
        # 从扩展ψ恢复速度
        psi_ext = F.pad(psi_ext.view(1,1,H,W),(1,1,1,1),mode="replicate")
        u_ext =  (psi_ext[:,:,2:,1:-1]-psi_ext[:,:,:-2,1:-1])/2
        v_ext = -(psi_ext[:,:,1:-1,2:]-psi_ext[:,:,1:-1,:-2])/2
        
        for _ in range(5):
            u_ext = F.avg_pool2d(u_ext, kernel_size=5, stride=1, padding=2)
            v_ext = F.avg_pool2d(v_ext, kernel_size=5, stride=1, padding=2)
            u_ext[:, :, self.h:2*self.h, self.w:2*self.w] = u
            v_ext[:, :, self.h:2*self.h, self.w:2*self.w] = v

        return u_ext, v_ext # shape 1*1*H*W

        
    def evaluate(self, x_, y_):
        """
        查询任意 float 坐标点上的速度 (u,v)，x_,y_ 为一维数组
        """
        if x_.dim() == 1:
            x_ = x_.unsqueeze(0)
        if y_.dim() == 1:
            y_ = y_.unsqueeze(0)
        
        xq_norm =  2*(x_+self.h)/(3*self.h-1)-1
        yq_norm =  2*(y_+self.w)/(3*self.w-1)-1
        grid = torch.stack((yq_norm, xq_norm), dim=-1).unsqueeze(0)
        grid = grid.to(self.device)

        u_val = F.grid_sample(self.u_ext, grid, mode='bilinear', align_corners=True)
        v_val = F.grid_sample(self.v_ext, grid, mode='bilinear', align_corners=True)
        return v_val.squeeze(), u_val.squeeze()


class AnalyticalFlow:
    def __init__(self, name="uniform", **kwargs):
        self.name = name
        self.params = kwargs
        self.flow_fn = self._get_flow_fn(name)

    def _get_flow_fn(self, name):
        if name == "uniform":
            return self._uniform
        elif name == "solid_rot":
            return self._solid_rot
        elif name == "lamb_oseen":
            return self._lamb_oseen
        elif name == "sin_flow":
            return self._sin_flow
        elif name == "cellular_flow":
            return self._cellular_flow
        elif name == "shear_flow":
            return self._shear_flow
        elif name =="gs_flow":
            return self._gs_flow
        elif name == "one_shear":
            return self._one_sided_shear_flow
        else:
            raise ValueError(f"Unsupported flow type: {name}")

    def evaluate(self, x, y):
        u,v = self.flow_fn(x, y, **self.params)
        return u,v

    # 均匀流动
    def _uniform(self, x, y, uc=0.2, vc=0.0):
        u = torch.full(x.shape, uc, dtype=torch.float32, device=x.device)
        v = torch.full(x.shape, vc, dtype=torch.float32, device=x.device)
        return u, v
    
    def _solid_rot(self, x, y, x_c, y_c, omega=0.06):
        x, y = x - x_c, y - y_c
        r = torch.sqrt(x**2 + y**2)
        theta = torch.arctan2(y, x)
        u, v = -r * omega * torch.sin(theta), r * omega * torch.cos(theta)
        return u, v

    def _one_sided_shear_flow(self, x, y, vmax=10, vmin=0.0, p=256):
        t = y / p
        u = vmax * (1 - t) + vmin * t
        v = torch.zeros_like(u)

        return u, v
    
    def _lamb_oseen(self, x, y, x_c, y_c, Gamma=5e3, rc=40):
        x, y = x - x_c, y - y_c
        r = torch.sqrt(x**2 + y**2) + 1e-8
        theta = torch.arctan2(y, x)
        Amp = Gamma * (1 - torch.exp(-r**2 / rc**2)) / (2 * torch.pi * r)
        u, v = -Amp * torch.sin(theta), Amp * torch.cos(theta)
        return u, v
    
    def _sin_flow(self, x, y, x_c, y_c, a=6, b=128, scale=5):
        x, y = x - x_c, y - y_c
        theta = torch.arctan(a * torch.cos(2 * torch.pi * x / b) * 2 * torch.pi / b)
        u, v = scale * torch.cos(theta), scale * torch.sin(theta)
        return u, v
    
    def _cellular_flow(self, x, y, vmax=10, p=128):
        u = vmax*torch.sin(2*torch.pi*x/p)*torch.cos(2*torch.pi*y/p)
        v = -vmax*torch.cos(2*torch.pi*x/p)*torch.sin(2*torch.pi*y/p)
        return u, v

    def _shear_flow(self, x, y, vmax=10, p=256):
        vmin=vmax/5
        # 到中心线的归一化距离 [0,1]
        dist = torch.abs(y - p/2) / (p/2)
    
        # 线性插值：中心 vmin，边界 vmax
        u = vmin + (vmax - vmin) * dist
    
        v = torch.zeros_like(u)
    
        return u, v


    def _gs_flow(self, x, y, max_v=5.0, n_vortex=9, seed=42, H=256, W=256):
        rng = np.random.default_rng(seed)
    
        n_cols = int(np.round(np.sqrt(n_vortex)))
        n_rows = (n_vortex + n_cols - 1) // n_cols
        margin = H // (n_cols * 2)
        ys = np.linspace(margin, H - margin, n_rows)
        xs = np.linspace(margin, W - margin, n_cols)
        grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
        cy = grid_y.reshape(-1)[:n_vortex] + rng.uniform(-margin*0.3, margin*0.3, n_vortex)
        cx = grid_x.reshape(-1)[:n_vortex] + rng.uniform(-margin*0.3, margin*0.3, n_vortex)
        centers = torch.tensor(np.stack([cy, cx], axis=1),
                               dtype=torch.float32, device=x.device)
    
        sigma_base = float(H / n_cols) * 0.6
        sigmas = torch.tensor(
            rng.uniform(sigma_base * 0.8, sigma_base * 1.2, n_vortex),
            dtype=torch.float32, device=x.device
        )
        signs  = np.array([(1 if (i//n_cols + i%n_cols) % 2 == 0 else -1)
                           for i in range(n_vortex)])
        gammas = torch.tensor(
            signs * rng.uniform(0.7, 1.0, n_vortex),
            dtype=torch.float32, device=x.device
        )
    
        u = torch.zeros_like(x)
        v = torch.zeros_like(x)
        for i in range(len(gammas)):
            dy   = y - centers[i, 0]
            dx   = x - centers[i, 1]
            r2   = dx**2 + dy**2
            sig2 = sigmas[i]**2 + 1e-6
            # 直接用高斯定义旋转速度，峰值在涡心，向外平滑衰减
            gauss = torch.exp(-r2 / (2 * sig2))
            u += gammas[i] * (-dy) * gauss
            v += gammas[i] * ( dx) * gauss
    
        mag_max = torch.sqrt(u**2 + v**2).max()
        u = u / (mag_max + 1e-6) * max_v
        v = v / (mag_max + 1e-6) * max_v
        return u, v


    

class FlowManager:
    def __init__(self, mode="data", **kwargs):
        if mode == "data":
            self.field = FlowField(**kwargs)
        elif mode == "analytical":
            self.field = AnalyticalFlow(**kwargs)
        else:
            raise ValueError("mode must be 'data' or 'analytical'")
        self.mode = mode

    def evaluate(self, x_, y_):
        return self.field.evaluate(x_, y_)