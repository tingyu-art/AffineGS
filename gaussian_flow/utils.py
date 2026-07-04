import torch
import torch.nn.functional as F


# Sobel kernels are module-level constants moved to the target device on demand.
_SOBEL_W = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3).expand(2, -1, -1, -1)
_SOBEL_H = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3).expand(2, -1, -1, -1)


def _has_jacobian(model) -> bool:
    """检查模型是否启用了雅可比矩阵（J_params 非 None）。"""
    return getattr(model, 'enable_jacobian', False) and model.J_params is not None


def build_event_density_map(events, H, W, smooth_k=9, device='cuda'):
    """
    构建平滑事件密度图 [H, W]。
    events[:,0]=h（行坐标），events[:,1]=w（列坐标）。
    输出在指定 device 上。
    """
    events_map = torch.zeros(H, W, device=device)
    hs = events[:, 0].long().clamp(0, H - 1)
    ws = events[:, 1].long().clamp(0, W - 1)
    events_map.index_put_((hs, ws), torch.ones(len(events), device=device), accumulate=True)
    return F.avg_pool2d(events_map[None, None], smooth_k, stride=1,
                        padding=smooth_k // 2).squeeze()


def flow_regularizer(flow, eps=1e-6, div_q=0.8, tau_detach=True):
    u = flow.unsqueeze(0)  # [1, 2, H, W]
    u_pad = F.pad(u, (1, 1, 1, 1), mode='replicate')

    du_dh = 0.5 * (u_pad[:, 0:1, 2:,   1:-1] - u_pad[:, 0:1, :-2,  1:-1])
    dv_dw = 0.5 * (u_pad[:, 1:2, 1:-1, 2:  ] - u_pad[:, 1:2, 1:-1, :-2 ])

    div = du_dh + dv_dw
    abs_div = div.abs()

    tau = torch.quantile(abs_div.detach(), div_q)
    excess_div = torch.clamp(abs_div - tau, min=0.0)
    loss_div = torch.sqrt(excess_div ** 2 + eps ** 2).mean()

    d2u_dhh = u_pad[:, 0:1, 2:,   1:-1] - 2.0 * u_pad[:, 0:1, 1:-1, 1:-1] + u_pad[:, 0:1, :-2,  1:-1]
    d2u_dww = u_pad[:, 0:1, 1:-1, 2:  ] - 2.0 * u_pad[:, 0:1, 1:-1, 1:-1] + u_pad[:, 0:1, 1:-1, :-2 ]

    d2v_dhh = u_pad[:, 1:2, 2:,   1:-1] - 2.0 * u_pad[:, 1:2, 1:-1, 1:-1] + u_pad[:, 1:2, :-2,  1:-1]
    d2v_dww = u_pad[:, 1:2, 1:-1, 2:  ] - 2.0 * u_pad[:, 1:2, 1:-1, 1:-1] + u_pad[:, 1:2, 1:-1, :-2 ]

    loss_smooth = torch.sqrt(
        d2u_dhh**2 + d2u_dww**2 + d2v_dhh**2 + d2v_dww**2 + eps**2
    ).mean()

    return loss_smooth, loss_div

