import os
import math
import json
import torch
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# ====================== 全局配置 ======================
plt.rcParams['font.family'] = "Arial"  # Arial, Times New Roman,  SimHei, STSONG
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['pdf.fonttype'] = 42

cmap = mpl.colormaps['tab10']
colors = cmap(np.linspace(0, 1, 10))
colors = np.concatenate([colors]*10, axis=0)
markers = ["s", "o", "^", "P", "*", "v", "<", "X", ">", "D"]*10


def extract_data(origin_data):

    data = {key: value for key, value in origin_data.items() if key != 'uniform_flow'}
    
    # 假设data是已加载的数据结构
    x_vals = []
    y_rmse_list = []  # 存储所有流场的rmse数据
    y_aee_list = []   # 存储所有流场的aee数据
    
    for key, values in data.items():
        current_x = []
        current_rmse = []
        current_aee = []
        
        for param, metrics in values.items():
            param_value = float(param)
            current_x.append(param_value)
            current_rmse.append(metrics.get("rmse", 0))
            current_aee.append(metrics.get("aee", 0))
        
        # 按参数值排序
        sorted_indices = np.argsort(current_x)
        x_sorted = np.array(current_x)[sorted_indices]
        rmse_sorted = np.array(current_rmse)[sorted_indices]
        aee_sorted = np.array(current_aee)[sorted_indices]
        
        x_vals = x_sorted
        y_rmse_list.append(rmse_sorted)
        y_aee_list.append(aee_sorted)

    return x_vals, y_rmse_list, y_aee_list

def plot_curvegraph(x_data, y_data_list, legend_names_en, legend_names_zh,
    title_en, title_zh, xlabel_en, xlabel_zh, ylabel_en, ylabel_zh,
    ticks_size=None, legend_size=None, label_size=None, 
    xlim=None, ylim=None, 
    x_major_ticks=None, x_minor_ticks=None,
    y_major_ticks=None, y_minor_ticks=None,
    x_log=False, y_log=False,
    filename=None, use_chinese=False, figsize=(5, 3), 
    colors=colors, markers=markers
):

    # 设置语言内容
    title = title_zh if use_chinese else title_en
    xlabel = xlabel_zh if use_chinese else xlabel_en
    ylabel = ylabel_zh if use_chinese else ylabel_en
    legends = legend_names_zh if use_chinese else legend_names_en

    # 中文字体设置
    if use_chinese:
        plt.rcParams['font.family'] = ['STSONG']

    fig, ax = plt.subplots(figsize=figsize)

    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # X轴设置
    if xlim:
        ax.set_xlim(xlim)

    if x_log:
        ax.set_xscale('log')
        if xlim:
            xmin, xmax = xlim
            min_exp = math.floor(math.log10(xmin))
            max_exp = math.ceil(math.log10(xmax))

            # 主刻度：10^1, 10^2, ...
            major_ticks = [10 ** exp for exp in range(min_exp, max_exp + 1)]
            ax.xaxis.set_major_locator(plt.FixedLocator(major_ticks))
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f'$10^{{{int(round(math.log10(x)))}}}$')
            )

            # 副刻度：指数间隔 0.2（log 均匀）
            minor_exp = np.arange(min_exp, max_exp + 0.01, 0.2)
            minor_ticks = [10 ** exp for exp in minor_exp]

            minor_ticks = [t for t in minor_ticks if xmin <= t <= xmax]
            ax.xaxis.set_minor_locator(plt.FixedLocator(minor_ticks))

    else:
        if x_major_ticks:
            ax.xaxis.set_major_locator(plt.FixedLocator(x_major_ticks))
        if x_minor_ticks:
            ax.xaxis.set_minor_locator(plt.FixedLocator(x_minor_ticks))

    # Y轴设置
    if ylim:
        ax.set_ylim(ylim)

    if y_log:
        ax.set_yscale('log')
        if ylim:
            ymin, ymax = ylim
            min_exp = math.floor(math.log10(ymin))
            max_exp = math.ceil(math.log10(ymax))

            # 主刻度：10^1, 10^2, ...
            major_ticks = [10 ** exp for exp in range(min_exp, max_exp + 1)]
            ax.yaxis.set_major_locator(plt.FixedLocator(major_ticks))
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f'$10^{{{int(round(math.log10(y)))}}}$')
            )

            # 副刻度：指数间隔 0.2（log 均匀）
            minor_exp = np.arange(min_exp, max_exp + 0.01, 0.2)
            minor_ticks = [10 ** exp for exp in minor_exp]

            minor_ticks = [t for t in minor_ticks if ymin <= t <= ymax]
            ax.yaxis.set_minor_locator(plt.FixedLocator(minor_ticks))

    else:
        if y_major_ticks:
            ax.yaxis.set_major_locator(plt.FixedLocator(y_major_ticks))
        if y_minor_ticks:
            ax.yaxis.set_minor_locator(plt.FixedLocator(y_minor_ticks))

    ax.tick_params(axis='both', which='major', direction='out', length=6, width=1, labelsize=ticks_size)
    ax.tick_params(axis='both', which='minor', direction='out', length=3, width=1, labelsize=ticks_size)
    ax.grid(True, which='major', linestyle='-', linewidth=1)

    # 绘制曲线
    for i, y_data in enumerate(y_data_list):
        if len(x_data) != len(y_data):
            raise ValueError(f"x_data和y_data_list[{i}]的长度不一致")
    
        sorted_indices = np.argsort(x_data)
        x_sorted = np.array(x_data)[sorted_indices]
        y_sorted = np.array(y_data)[sorted_indices]
    
        color = colors[i % len(colors)] if colors is not None and len(colors) > 0 else None
        marker = markers[i % len(markers)] if markers is not None and len(markers) > 0 else None
    
        ax.plot(x_sorted, y_sorted,
                marker=marker,
                color=color,
                label=legends[i],
                markersize=6,
                linewidth=1.5)

    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=legend_size, loc=1, framealpha=0.6)

    plt.tight_layout()

    # 文件保存逻辑
    if filename:
        base, ext = os.path.splitext(filename)
        if use_chinese:
            filename = f"{base}-zh{ext}"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


        
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


def _infer_dataset_name(file_path):
    """
    根据输入文件路径自动提取数据集名称。
    默认取文件名（去掉扩展名）。
    例如:
        /xxx/JHTDB.json   -> JHTDB
        /xxx/dns_flow.json -> dns_flow
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def plot_boxplot(
    file_path,
    metric='rmse',
    save_path=None,
    ylabel_en=None,
    ylabel_zh=None,
    xlabel_en='Methods',
    xlabel_zh='方法',
    title_en=None,
    title_zh=None,
    box_width=0.6,
    y_lim=None,
    label_size=12,
    tick_size=10,
    grid=True,
    use_chinese=False,
    dataset_name=None,          # 新增：可手动指定图例名称
    legend_loc='upper right',   # 新增：图例位置
    legend_line_color='black',  # 新增：图例横线颜色
    legend_line_width=1.5,      # 新增：图例横线宽度
):
    # ===================== 读取 JSON =====================
    with open(file_path, 'r', encoding='utf-8') as f:
        final_results = json.load(f)

    # 你的方法顺序（必须和计算代码一致）
    method_names = ["ILK_VG", "OF_VG", "INR_EST", "INR_VG", "ConstGS", "AffineGS"]
    display_names = ['ILK-VG', 'OF-VG', 'INR-EST', 'INR-VG', 'ConstGS', 'AffineGS']

    # ===================== 正确提取数据 =====================
    data = []
    all_values = []

    for method in method_names:
        values = []
        for file_key in final_results:
            metric_val = final_results[file_key][method][metric]
            values.append(metric_val)

        data.append(values)
        all_values.extend(values)

    # ===================== 自动获取数据集名称 =====================
    if dataset_name is None:
        dataset_name = _infer_dataset_name(file_path)

    # ===================== 字体设置 =====================
    plt.rcParams['font.family'] = "Arial"
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': label_size,
        'axes.titlesize': 12,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'legend.fontsize': 10,
        'axes.unicode_minus': False
    })

    if use_chinese:
        plt.rcParams['font.family'] = ['STSONG']

    # ===================== 绘制箱线图 =====================
    fig, ax = plt.subplots(figsize=(5, 3))

    sns.boxplot(
        data=data,
        width=box_width,
        showfliers=False,
        boxprops=dict(facecolor='none', edgecolor='black', linewidth=1.0),
        medianprops=dict(color='orange', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.0),
        capprops=dict(color='black', linewidth=1.0),
        ax=ax
    )

    # 边框与刻度
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(axis='both', width=1.0, length=6)

    # 网格
    if grid:
        ax.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.7)

    # X轴标签
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(display_names, rotation=0, ha='center')

    # 标题/轴标签
    xlabel = xlabel_zh if use_chinese else xlabel_en
    ylabel = ylabel_zh if use_chinese else ylabel_en
    title = title_zh if use_chinese else title_en

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # Y轴范围
    if y_lim:
        ax.set_ylim(y_lim)
    elif all_values:
        min_val, max_val = min(all_values), max(all_values)
        range_val = max_val - min_val
        if range_val == 0:
            ax.set_ylim(min_val - 0.1, max_val + 0.1)
        else:
            ax.set_ylim(min_val - 0.1 * range_val, max_val + 0.1 * range_val)

    # ===================== 增加图例 =====================
    legend_handle = Line2D(
        [0], [0],
        color=legend_line_color,
        lw=legend_line_width,
        linestyle='-'
    )
    ax.legend(
        handles=[legend_handle],
        labels=[dataset_name],
        loc=legend_loc,
        frameon=True,           # 打开边框
        fancybox=True,          # 圆角框，和你图里类似
        framealpha=1.0,         # 不透明
        edgecolor='0.75',       # 边框颜色
        facecolor='white',      # 框内背景色
        handlelength=2.0,
        handletextpad=0.6,
        borderaxespad=0.4
    )

    plt.tight_layout()

    # 保存
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        print(f"✅ 箱线图已保存：{save_path}")

    plt.show()


def plot_vortex_line_panel(
    omega,                          # [C, H, W]，第0通道为真值
    axis='h',
    coord_idx=None,

    # ── 剖面图 y 轴（固定范围，不再自动调整）──────────────────
    yticks=[-1, -0.5, 0, 0.5, 1],  # 直接传刻度列表
    ylim=None,                      # None → 由 yticks 首尾推断

    # ── colorbar ─────────────────────────────────────────────────
    cbar_ticks=[-2, -1, 0, 1, 2],

    # ── 颜色 ─────────────────────────────────────────────────────
    colors=None,

    # ── 尺寸 & 间距（全部单位：inch）────────────────────────────
    panel_h=3,          # 每个方形面板的边长 [inch]（所有数据子图共用）
    img_profile_gap=1.2,  # 涡度图（含colorbar）→ 第1剖面图 的间距
    profile_gap=0.2,     # 剖面图彼此之间的间距
    cbar_w=0.15,          # colorbar 宽度 [inch]
    cbar_pad=0.08,        # colorbar 与涡度图的间距
    left_margin=0.55,     # 左侧白边（含 y 轴标签）
    right_margin=0.15,    # 右侧白边
    bottom_margin=0.50,   # 下方白边（含 x 轴标签）
    top_margin=0.15,      # 上方白边

    save_path=None,
    dpi=300,
):
    """
    所有数据子图（涡度图 + 三个剖面图）均为精确正方形，边长 = panel_h 英寸。
    通过 set_box_aspect(1) 强制 axes 外框纵横比为 1，无视数据范围。
    """
    # ── 1. 输入处理 ─────────────────────────────────────────────
    estimators = ['Ground truth', 'ILK_VG', 'OF_VG',
                  'INR_EST', 'INR_VG', 'ConstGS', 'AffineGS']

    omega_np = (omega.detach().cpu().numpy()
                if torch.is_tensor(omega) else np.asarray(omega))
    assert omega_np.ndim == 3, "omega must be [C, H, W]"
    C, H, W = omega_np.shape
    assert C == len(estimators), f"Expected {len(estimators)} channels, got {C}"

    groups = [{'indices': [1, 2]}, {'indices': [3, 4]}, {'indices': [5, 6]}]

    # ── 2. 截线位置 ─────────────────────────────────────────────
    if coord_idx is None:
        coord_idx = H // 2 if axis == 'h' else W // 2

    if axis == 'h':
        coord_idx = int(np.clip(coord_idx, 0, H - 1))
        coord = np.arange(W)
        xlabel = 'x $[px]$'
    elif axis == 'v':
        coord_idx = int(np.clip(coord_idx, 0, W - 1))
        coord = np.arange(H)
        xlabel = 'y $[px]$'
    else:
        raise ValueError("axis must be 'h' or 'v'")

    # ── 3. 提取剖面 ─────────────────────────────────────────────
    profiles = [
        omega_np[c, coord_idx, :] if axis == 'h' else omega_np[c, :, coord_idx]
        for c in range(C)
    ]
    gt_profile = profiles[0]
    gt_img     = omega_np[0]

    # ── 4. 颜色设置 ─────────────────────────────────────────────
    default_palette = ['#E07B39', '#5A9E6F', '#C94040',
                       '#7B68C8', '#3A7DC9', '#E0A020', '#2EAA8A', '#C4569A']
    if colors is None:
        colors = {}
    palette_iter = iter(default_palette)
    for name in estimators[1:]:
        if name not in colors:
            colors[name] = next(palette_iter)

    GT_STYLE = dict(color='black', lw=1.5, zorder=10, label='Ground truth')

    # ── 5. Tick 解析 ────────────────────────────────────────────
    _auto_vmax = float(np.max(np.abs(gt_img)))
    cbar_ticks = np.asarray(
        cbar_ticks if cbar_ticks is not None
        else np.round(np.linspace(-_auto_vmax, _auto_vmax, 7), 10),
        dtype=float)
    img_vmin, img_vmax = float(cbar_ticks[0]), float(cbar_ticks[-1])

    yticks = np.asarray(
        yticks if yticks is not None
        else np.round(np.linspace(-_auto_vmax, _auto_vmax, 7), 10),
        dtype=float)
    _ylim = tuple(ylim) if ylim is not None else (float(yticks[0]), float(yticks[-1]))

    if len(coord) >= 251:
        xticks = [0, 50, 100, 150, 200, 250]
    else:
        xticks = np.unique(
            np.round(np.linspace(coord[0], coord[-1], 6)).astype(int)
        ).tolist()

    # ── 6. 精确 inch 布局（所有数据子图均为 panel_h × panel_h）────
    # 图总宽 = 所有内容宽度 + 边距
    fig_w = (left_margin
             + panel_h + cbar_pad + cbar_w   # 涡度图 + colorbar
             + img_profile_gap
             + 3 * panel_h + 2 * profile_gap  # 三个正方形剖面图
             + right_margin)
    fig_h = bottom_margin + panel_h + top_margin

    fig = plt.figure(figsize=(fig_w, fig_h))

    # inch → figure 归一化坐标
    def fx(x): return x / fig_w
    def fy(y): return y / fig_h

    pw  = fx(panel_h)   # 子图宽度（归一化）
    ph  = fy(panel_h)   # 子图高度（归一化）
    bot = fy(bottom_margin)

    # 各轴左边界
    x_img = fx(left_margin)
    x_cax = x_img + pw + fx(cbar_pad)
    x_p1  = x_cax + fx(cbar_w) + fx(img_profile_gap)
    x_p2  = x_p1  + pw + fx(profile_gap)
    x_p3  = x_p2  + pw + fx(profile_gap)

    ax_img = fig.add_axes([x_img, bot, pw, ph])
    cax    = fig.add_axes([x_cax, bot, fx(cbar_w), ph])
    ax_p1  = fig.add_axes([x_p1,  bot, pw, ph])
    ax_p2  = fig.add_axes([x_p2,  bot, pw, ph])
    ax_p3  = fig.add_axes([x_p3,  bot, pw, ph])
    profile_axes = [ax_p1, ax_p2, ax_p3]

    # ── 关键：强制所有数据子图的 axes 外边框为正方形 ─────────────
    for ax in [ax_img, ax_p1, ax_p2, ax_p3]:
        ax.set_box_aspect(1)   # 物理纵横比为 1

    label_fs  = 12
    tick_fs   = 12
    legend_fs = 11

    # ── 7. 第1列：真值涡度图 ───────────────────────────────────
    im = ax_img.imshow(gt_img, cmap='RdBu_r',
                       vmin=img_vmin, vmax=img_vmax,
                       origin='upper', interpolation='nearest')

    ax_img.set_xlabel('x $[px]$', fontsize=label_fs)
    ax_img.set_ylabel('y $[px]$', fontsize=label_fs)
    ax_img.tick_params(labelsize=tick_fs, direction='out', length=3.5, width=0.8)

    line_color = '#ff5a4f'
    if axis == 'h':
        ax_img.axhline(coord_idx, color=line_color, ls='--', lw=1.6, alpha=0.95,
                       label=f'Select pos (row={coord_idx})')
    else:
        ax_img.axvline(coord_idx, color=line_color, ls='--', lw=1.6, alpha=0.95,
                       label=f'Select pos (col={coord_idx})')

    ax_img.legend(fontsize=legend_fs, loc='upper right', frameon=True,
                  framealpha=0.92, borderpad=0.3, handlelength=1.6, handletextpad=0.5)

    # ── 8. Colorbar ──────────────────────────────────────────────
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks(cbar_ticks)
    cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    cb.ax.tick_params(labelsize=tick_fs, direction='out', length=2.5, width=0.7, pad=2)

    # ── 9. 剖面图（严格使用固定的 yticks/ylim）──────────────────
    for col, (ax, grp) in enumerate(zip(profile_axes, groups)):
        ax.plot(coord, gt_profile, **GT_STYLE)
        for idx in grp['indices']:
            name = estimators[idx]
            ax.plot(coord, profiles[idx], color=colors[name], lw=1.5, label=name)

        ax.set_xlim(coord[0], coord[-1])
        ax.set_ylim(_ylim)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.8, zorder=0)
        ax.grid(True, lw=0.5, alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=label_fs)
        ax.tick_params(labelsize=tick_fs, direction='out', length=3.5, width=1)

        if col == 0:
            ax.set_ylabel('Vorticity', fontsize=label_fs)
        else:
            ax.tick_params(axis='y', left=False, labelleft=False)

        ax.legend(fontsize=legend_fs, loc='upper right', frameon=True,
                  framealpha=0.92, borderpad=0.25, handlelength=1.6,
                  handletextpad=0.5, labelspacing=0.25)

    # ── 10. 统一边框线宽 ─────────────────────────────────────────
    for ax in [ax_img, cax, *profile_axes]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter, MaxNLocator

def plot_exp111(data, flow_names, metric="rmse", save_path=None, y_scale="auto"):
    assert metric in ["rmse", "aee"], "metric 必须是 rmse 或 aee"

    flow_keys  = ["flow0", "flow1", "flow2"]
    style_names = {"1": "w-J", "0": "w/o-J"}

    series = []
    all_x = []
    all_y = []
    for i, flow_key in enumerate(flow_keys):
        assert flow_key in data, f"{flow_key} 不存在"
        flow_data = data[flow_key]
        for jac_key in ["1", "0"]:
            assert jac_key in flow_data, f"{flow_key} 缺少 jacobian={jac_key}"
            sub_data = flow_data[jac_key]
            xs = np.array(sorted([int(k) for k in sub_data.keys()]), dtype=float)
            ys = np.array([sub_data[str(int(x))][metric] for x in xs], dtype=float)
            series.append((i, jac_key, xs, ys))
            all_x.extend(xs.tolist())
            all_y.extend(ys.tolist())

    all_x = np.array(all_x, dtype=float)
    all_y = np.array(all_y, dtype=float)
    valid_y = all_y[np.isfinite(all_y) & (all_y > 0)]
    if valid_y.size == 0:
        raise ValueError(f"{metric} 没有可绘制的正数数据")

    y_min = float(valid_y.min())
    y_max = float(valid_y.max())
    data_ratio = y_max / max(y_min, 1e-12)
    use_log_y = (y_scale == "log") or (y_scale == "auto" and data_ratio >= 5.0)

    fig, ax = plt.subplots(figsize=(5, 3))

    # ===== 主曲线 =====
    for i, jac_key, xs, ys in series:
        linestyle = "-" if jac_key == "1" else "--"
        ax.plot(
            xs,
            ys,
            linestyle=linestyle,
            marker=markers[i],
            color=colors[i],
            linewidth=1.8,
            markersize=5.4,
            markeredgewidth=0.6,
            markeredgecolor="white",
        )

    # ===== 坐标轴 =====
    ax.set_xlabel("Number of Gaussians (init_n)", fontsize=12)
    ax.set_ylabel(f"{metric.upper()} $[px]$", fontsize=12)

    x_pad = max(10.0, 0.03 * (all_x.max() - all_x.min()))
    ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

    if use_log_y:
        ax.set_yscale("log")
        ax.set_ylim(y_min / 1.22, y_max * 1.22)
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9)))

        def y_major_formatter(y, pos):
            if y <= 0:
                return ""
            exp = np.log10(y)
            if abs(exp - round(exp)) < 1e-10:
                return rf"$10^{{{int(round(exp))}}}$"
            return ""

        ax.yaxis.set_major_formatter(FuncFormatter(y_major_formatter))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.45)
        ax.grid(True, which="minor", axis="y", linestyle="--", alpha=0.22)
    else:
        y_pad = max(0.02, 0.08 * (y_max - y_min))
        ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, which="major", axis="both", linestyle="--", alpha=0.45)

    ax.tick_params(labelsize=10, direction="out", length=3.5, width=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

    # ===== 图例：拆成两个独立图例 =====
    color_handles = [
        mlines.Line2D([], [], color=colors[0], marker=markers[0], linestyle='None',
                      markersize=6, label=flow_names[0]),
        mlines.Line2D([], [], color=colors[1], marker=markers[1], linestyle='None',
                      markersize=6, label=flow_names[1]),
        mlines.Line2D([], [], color=colors[2], marker=markers[2], linestyle='None',
                      markersize=6, label=flow_names[2]),
    ]

    style_handles = [
        mlines.Line2D([], [], color='black', linestyle='-',  linewidth=1.8, label=style_names["1"]),
        mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.8, label=style_names["0"]),
    ]

    # 保持原来的双图例布局：右上角是三个流场标签，偏左是模块开关标签。
    legend1 = ax.legend(
        handles=color_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=1,
        frameon=True,
        fontsize=9,
        borderpad=0.3,
        handletextpad=0.5,
        labelspacing=0.35,
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=style_handles,
        loc="upper right",
        bbox_to_anchor=(0.65, 1.02),
        frameon=True,
        fontsize=9,
        borderpad=0.3,
        handlelength=1.8,
        handletextpad=0.6,
        labelspacing=0.35,
    )

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def analyze_error(pred_flow, gt_flow, flow_names, n_bins=5):
    """
    按真值速度百分比区间，统计每个速度区间的平均EPE。
    pred_flow, gt_flow: [C, 2, H, W]
    """
    C, _, H, W = pred_flow.shape

    # 速度模长
    gt_mag_all = gt_flow.norm(dim=1)  # [C, H, W]

    # 所有通道共用速度区间
    v_min = gt_mag_all.min().item()
    v_max = gt_mag_all.max().item()
    bins = torch.linspace(v_min, v_max, n_bins + 1)

    avg_epe_per_channel = [[] for _ in range(C)]

    for i in range(n_bins):
        lo = bins[i].item()
        hi = bins[i+1].item()

        for c in range(C):
            gt_mag = gt_mag_all[c]
            pred = pred_flow[c]
            epe_map = (pred - gt_flow[c]).norm(dim=0)  # [H, W]
            mask = (gt_mag >= lo) & (gt_mag < hi)
            if mask.sum() == 0:
                avg_epe = 0.0
            else:
                avg_epe = epe_map[mask].mean().item()
            avg_epe_per_channel[c].append(avg_epe)

    # 横轴标签：百分比区间
    x_labels = [f"{int(i*100/n_bins)}~{int((i+1)*100/n_bins)}%" for i in range(n_bins)]
    x = range(n_bins)

    # ================= 图1：EPE vs 速度百分比 =================
    plt.figure(figsize=(5, 3))
    for c in range(C):
        plt.plot(x, avg_epe_per_channel[c], marker=markers[c], label=f'Channel {c}')

    plt.xticks(x, x_labels)
    plt.xlabel("Velocity magnitude (%)")
    plt.ylabel("Average EPE")
    plt.legend(flow_names, loc = "upper right")
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.show()

# ================= 图2：训练损失曲线 =================
def plot_loss_curves(history, flow_names, num_curves=3, save_path=None):
    """
    history: 包含 'total' 键的字典列表，或直接是损失值列表
    num_curves: 要绘制的曲线数量（取前几个）
    """
    loss_curves = [h['total'] for h in history[:num_curves]]
    
    plt.figure(figsize=(5, 3))
    
    for i, loss in enumerate(loss_curves):
        # 如果 loss 是标量，转为 list
        if isinstance(loss, (int, float)):
            loss = [loss]
        plt.plot(range(len(loss)), loss, linewidth=1.5, label=f'Run {i+1}')
    
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Total loss", fontsize=12)
    plt.legend(flow_names)
    plt.xlim([-5, 305])
    plt.ylim([-9, -3])
    plt.grid(True, alpha=0.8)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
