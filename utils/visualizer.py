# visualizer.py
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.ticker import FuncFormatter


def plot_piv_image_pair(img1, img2, output_file=None):
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
        
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)           # 1行2列，第一张图
    plt.imshow(img1, cmap='gray')
    plt.title('Image 1')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)           # 1行2列，第二张图
    plt.imshow(img2, cmap='gray')
    plt.title('Image 2')
    plt.axis('off')

    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to {output_file}")
    plt.show()


def save_piv_video(image_seq, output_file="output/output.mp4", axis=0, fps=None):
    """
    将生成的图像帧序列导出为 MP4 视频文件。
    
    参数：
    - output_file: 输出视频文件路径（默认 "output.mp4"）
    
    要求：
    - self.imgs: 已生成的图像帧列表，形状为 [H, W]，灰度图
    """
    if isinstance(image_seq, torch.Tensor):
        image_seq = image_seq.cpu().detach().numpy()
            
    fps = int(len(image_seq)/15) if fps is None else fps  # 帧率，可根据模拟需要调整

    if len(image_seq) == 0:
        raise ValueError("图像帧未生成")

    # 自动获取帧大小（支持任意宽高）
    frame_height, frame_width = image_seq[0].shape[:2]
    video_size = (frame_width, frame_height)

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, video_size)

    for frame in image_seq:
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().detach().numpy()

        # 确保类型和通道数为 uint8 & RGB
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)  # 灰度转RGB

        video_writer.write(frame)

    video_writer.release()
    print(f"[INFO] 视频保存至 {output_file}")

def plot_events_3d(events, output_file=None, startN=-10000, endN=-1):
    x = np.asarray(events[startN:endN,0])
    y = np.asarray(events[startN:endN,1])
    t = np.asarray(events[startN:endN,2])
    p = np.asarray(events[startN:endN,3])


    def expand_range(data, ratio=0.05):
        d_min, d_max = np.min(data), np.max(data)
        d_range = d_max - d_min
        if d_range == 0:
            d_range = 1e-6  # 防止为0时仍然是线
        return d_min - ratio * d_range, d_max + ratio * d_range
    
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x, y, t, marker='.', c=p)
    # 自动扩展三个坐标轴范围
    # ax.set_xlim(*expand_range(x))
    # ax.set_ylim(*expand_range(y))
    # ax.set_zlim(*expand_range(t))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('T')
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1)
        print(f"Figure saved to {output_file}")
    plt.show()


def plot_event_time2d(
    events,
    output_file=None,
    vmin=None,
    vmax=None,
    figsize=(5, 3),
    cmap='jet',
    cbar_label='Time (ms)',
    cbar_ticks=None,
    cbar_tick_size=12,
    cbar_label_size=12,
    image_rect=(0.02, 0.08, 0.84, 0.84),
    cbar_rect=(0.89, 0.08, 0.025, 0.84),
):
    """
        events: numpy.ndarray, 形状 (N,4)，列为 x, y, t, p
        vmin, vmax: 时间映射的颜色范围（可选）
    """
    # 分割事件数组
    if isinstance(events,torch.Tensor):
        events = events.cpu().detach().numpy()
    
    x, y, t, p = np.split(events, 4, axis=1)
    x = x.flatten().astype(int)
    y = y.flatten().astype(int)
    t = t.flatten()
    
    # 计算坐标偏移与图像大小
    x0, y0 = np.min(x), np.min(y)
    x_sz = np.max(x) - x0 + 1
    y_sz = np.max(y) - y0 + 1
    
    # 创建空图像，行对应y，列对应x
    img = np.zeros((x_sz, y_sz))+0.0
    
    # 赋值：注意索引顺序是 img[y, x]
    for xi, yi, ti in zip(x, y, t):
        yi_idx = yi - y0
        xi_idx = xi - x0
        img[xi_idx, yi_idx] = max(img[xi_idx, yi_idx], ti)  # 同点多事件取最大时间
    
    # 设置vmin/vmax默认值
    if vmin is None:
        vmin = np.min(t)
    if vmax is None:
        vmax = np.max(t)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(image_rect)
    cax = fig.add_axes(cbar_rect)

    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    ax.axis('off')

    cbar = fig.colorbar(im, cax=cax)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    cbar.set_label(cbar_label, fontsize=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_tick_size, direction='out', length=4, width=1)
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(1)
    
    if output_file is not None:
        fig.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"Figure saved to {output_file}")
    # plt.title('2D Event Time Visualization')
    plt.show()


def save_voxel_grid_video(events, output_file="output/voxelgrids.mp4",img_size=None, fps=24):
    def _event2voxelgrid3(events, img_size, num_bins=None, t_min=None, t_max=None):
        """
        将事件数据转换为3通道体素网格：正极性、负极性、总和。
    
        参数：
            events: np.ndarray, shape (N, 4)，每列为 (x, y, t, p)
            img_size: tuple (H, W)，图像空间尺寸
            num_bins: int，体素时间层数
            t_min: float, 时间最小值，None时使用events中最小t
            t_max: float, 时间最大值，None时使用events中最大t
    
        返回：
            voxel_grid: np.ndarray, shape (num_bins, H, W, 3)
        """
        if isinstance(events, torch.Tensor):
            events = events.cpu().detach().numpy()
        x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
        if img_size is None:
            img_size = np.max(x).astype(np.int64), np.max(y).astype(np.int64)
        H, W = img_size

        if len(events)>10000000:
            events = events[:10000000]
        num_bins = np.int64(len(events)/(H*W*0.32)) if num_bins is None else num_bins
    
        if t_min is None:
            t_min = t.min()
        if t_max is None:
            t_max = t.max()
        t_range = t_max - t_min
        if t_range == 0:
            t_range = 1e-9
    
        t_norm = (t - t_min) / t_range
        bin_idx = (t_norm * (num_bins - 1)).astype(int)
        bin_idx = np.clip(bin_idx, 0, num_bins - 1)
    
        # 初始化3个通道
        voxel_grid = np.zeros((num_bins, H, W, 3), dtype=np.float32)
    
        x_idx = x.astype(int)
        y_idx = y.astype(int)
    
        # 越界事件过滤
        mask = (x_idx >= 0) & (x_idx < H) & (y_idx >= 0) & (y_idx < W)
        x_idx = x_idx[mask]
        y_idx = y_idx[mask]
        bin_idx = bin_idx[mask]
        p = p[mask]
    
        for b, yi, xi, pol in zip(bin_idx, y_idx, x_idx, p):
            if pol > 0:
                voxel_grid[b, xi, yi, 2] += 1  # 正事件通道
            else:
                voxel_grid[b, xi, yi, 0] += 1  # 负事件通道
            # voxel_grid[b, xi, yi, 2] += pol   # 正负总和通道
        return voxel_grid
    # voxel_grid = _event2voxelgrid(events, img_size, num_bins=10, t_min=None, t_max=None)
    # print(len(voxel_grid))
    voxel_grid = _event2voxelgrid3(events, img_size, num_bins=None, t_min=None, t_max=None)
    voxel_grid_visual = np.clip(voxel_grid*128/(np.median(voxel_grid)+1),a_min=0,a_max=255)
    # voxel_grid_visual[:,:,:,1]
    mask = np.all(voxel_grid_visual == 0, axis=-1)
    voxel_grid_visual[mask] = 255
    print(voxel_grid_visual.shape, np.sum(voxel_grid))
    save_piv_video(voxel_grid_visual, output_file=output_file, axis=0, fps=fps)
    return True
    

def plot_flow_field(u, v, u_gt=None, v_gt=None, output_file=None, 
                    magFlag=True, slFlag=False, colorbar_flag=True,
                    step=1, quiver_step=16, v_min=None, v_max=None, 
                    scale=None, figsize=(5, 3), cbar_ticks=None, 
                    cbar_tick_size=12):
    """
    可视化速度场：
    - 背景：速度大小的热力图
    - 前景：流线图 + 稀疏箭头图（quiver）
    - 如果提供了u_gt, v_gt，计算并显示误差场（error magnitude）

    参数：
    - u, v: 速度场的两个分量，形状 [H, W]
    - u_gt, v_gt: ground truth 速度场，只有在需要计算误差时使用
    - magFlag: 是否显示速度大小热力图背景
    - slFlag: 是否显示流线图
    - colorbar_flag: 是否显示颜色条
    - step: 背景图采样间隔（默认=1，表示全分辨率）
    - quiver_step: 箭头稀疏采样间隔（默认=10）
    - cbar_ticks: 颜色条刻度设置
    """
    # 转换为numpy数组（如果输入是PyTorch张量）
    if isinstance(u, torch.Tensor):
        u = u.detach().cpu().numpy()
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()

    H, W = u.shape
    x = np.arange(0, H)
    y = np.arange(0, W)
    xx, yy = np.meshgrid(x, y, indexing='ij')

    # 计算速度大小
    magnitude = np.sqrt(u**2 + v**2)

    # 如果提供了 ground truth，计算误差场
    if u_gt is not None and v_gt is not None:
        if isinstance(u_gt, torch.Tensor):
            u_gt = u_gt.detach().cpu().numpy()
        if isinstance(v_gt, torch.Tensor):
            v_gt = v_gt.detach().cpu().numpy()
        error_magnitude = np.sqrt((u - u_gt)**2 + (v - v_gt)**2)
        display_data = error_magnitude
        data_label = 'Error Magnitude'
    else:
        display_data = magnitude
        data_label = 'Velocity Magnitude'

    plt.figure(figsize=figsize)

    # 背景热力图（速度大小或误差大小）
    if magFlag:
        # 统一处理 v_min 和 v_max 的设置
        if v_min is None:
            v_min = np.nanmin(display_data)
        if v_max is None:
            v_max = np.nanmax(display_data)

        im = plt.imshow(display_data[::step, ::step], vmin=v_min, vmax=v_max,
                        cmap='viridis', origin='lower')

        # 创建颜色条
        if colorbar_flag:
            cbar_ax = plt.gca().inset_axes([1.05, 0, 0.06, 1])  # 自定义位置和大小
            cbar = plt.colorbar(im, cax=cbar_ax)

            # 设置颜色条刻度
            if cbar_ticks is not None:
                cbar.set_ticks(cbar_ticks)
                
                # 计算需要显示的小数位数
                max_decimals = 0
                for tick in cbar_ticks:
                    if isinstance(tick, (int, float)):
                        # 将刻度值转换为字符串，检查小数位数
                        tick_str = f"{tick}"
                        if '.' in tick_str:
                            decimals = len(tick_str.split('.')[1])
                            max_decimals = max(max_decimals, decimals)
                
                # 使用固定的小数位数格式化所有刻度
                def format_with_decimals(x, pos):
                    return f"{x:.{max_decimals}f}"
                
                from matplotlib.ticker import FuncFormatter
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_with_decimals))
            else:
                # 如果没有提供刻度，使用默认格式
                def default_format(x, pos):
                    if abs(x) >= 10:
                        return f"{int(x)}"
                    elif abs(x) >= 1:
                        return f"{x:.1f}"
                    else:
                        return f"{x:.2f}"
                
                from matplotlib.ticker import FuncFormatter
                cbar.ax.yaxis.set_major_formatter(FuncFormatter(default_format))

            if cbar_tick_size is not None:
                cbar.ax.tick_params(labelsize=cbar_tick_size)
                
            # 设置颜色条标签
            # cbar.set_label(data_label, rotation=270, labelpad=15)
    else:
        plt.imshow(0 * display_data[::step, ::step], cmap='viridis', origin='lower')
        if colorbar_flag:
            plt.colorbar(label='')

    # 流线图（streamplot）
    if slFlag:
        plt.streamplot(yy, xx, v, u, color='white', linewidth=0.8, density=1.0)

    # 稀疏箭头图（quiver）
    plt.quiver(yy[quiver_step//2::quiver_step, quiver_step//2::quiver_step],
               xx[quiver_step//2::quiver_step, quiver_step//2::quiver_step],
               v[quiver_step//2::quiver_step, quiver_step//2::quiver_step],
               -u[quiver_step//2::quiver_step, quiver_step//2::quiver_step],
               color='red', scale=scale)

    # 轴和标题
    plt.xlabel("Y")
    plt.ylabel("X")
    plt.xlim([-0.6, W - 0.4])
    plt.ylim([-0.6, H - 0.4])
    plt.axis("off")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # 保存图片
    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')
        print(f"Figure saved to {output_file}")

    plt.show()
    

def create_colorbar(v_min, v_max, cmap='viridis', orientation='vertical', 
                   format_str="%.2f", figsize=None, output_file=None,
                   fontsize=12, label='EPE [px]'):
    """
    独立生成颜色条，保持一致的布局和1:3的宽高比
    
    参数：
    ◦ v_min: 颜色条的最小值
    ◦ v_max: 颜色条的最大值
    ◦ cmap: 颜色映射，默认为'viridis'
    ◦ orientation: 颜色条方向，'vertical'或'horizontal'
    ◦ format_str: 刻度格式字符串
    ◦ figsize: 图形尺寸（将自动设置为保持1:3宽高比）
    ◦ output_file: 输出文件路径（可选）
    ◦ fontsize: 字体大小
    ◦ label: 颜色条标签文本
    """
    # 确保v_min和v_max不同
    if np.isclose(v_min, v_max, atol=1e-5):
        v_max = v_min + 0.1
    
    # 设置固定宽高比1:3
    if orientation == 'vertical':
        if figsize is None:
            figsize = (1.2, 3.6)  # 1:3比例
        fig, ax = plt.subplots(figsize=figsize)
    else:
        if figsize is None:
            figsize = (3.6, 1.2)  # 1:3比例
        fig, ax = plt.subplots(figsize=figsize)
    
    # 创建归一化对象
    norm = colors.Normalize(vmin=v_min, vmax=v_max)
    
    # 创建ScalarMappable对象
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # 创建颜色条
    cbar = plt.colorbar(sm, cax=ax, orientation=orientation, format=format_str)
    
    # 确保刻度为6个
    tick_step = (v_max - v_min) / 5  # 分成6个刻度
    ticks = np.linspace(v_min, v_max, 6)  # 创建6个均匀分布的刻度

    # 格式化刻度标签，确保显示为需要的格式
    formatted_ticks = []
    for tick in ticks:
        if tick >= 10:
            formatted_ticks.append(f"{tick:.0f}")  # 格式化为整数
        else:
            formatted_ticks.append(f"{tick:.2f}")  # 保留两位小数

    # 设置刻度
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(formatted_ticks)
    
    # 设置字体大小
    cbar.ax.tick_params(labelsize=24)
    
    # 设置标签 - 只改变标签方向，其他保持不变
    if orientation == 'vertical':
        # 只改变标签文本方向，保持其他所有设置不变
        cbar.set_label(label, rotation=90, labelpad=2, fontsize=fontsize)
    else:
        cbar.set_label(label, fontsize=fontsize)
    
    # 调整布局以确保一致性
    plt.tight_layout(pad=1.0)
    
    # 保存或显示
    if output_file is not None:
        # 确保保存的图片保持1:3宽高比
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"Colorbar saved to {output_file}")
    else:
        plt.show()
    
    return fig, cbar
