import json
import math
import pathlib
import sys
import time
from collections import defaultdict
from itertools import groupby
sys.path.append("E:\Files\gitspace\\bbb-github")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brokenaxes import brokenaxes
from matplotlib.colors import to_rgb
from matplotlib.patches import (
    ConnectionPatch,
    Ellipse,
    FancyBboxPatch,
    Patch,
    PathPatch,
    Rectangle,
)
from matplotlib.path import Path
from matplotlib.ticker import PercentFormatter
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import gaussian_kde

SAVE_PREFIX = "E:\Files\A-blockchain\\branchbound\\branchbound仿真\\0129"
pathlib.Path.mkdir(pathlib.Path(SAVE_PREFIX), exist_ok=True)
SAVE = True

MAXSAT='maxsat'
TSP='tsp'
MIPLTP='miplib'

# 产生keyblock的方式
POW = "pow"
W_MINI = "withmini"

# openblock选择策略, 如open prblm不是BEST策略就先选block再选prblm
OB_SPEC = "ob_specific" # 默认选择第一个
OB_RAND = "ob_random"
OB_DEEP = "ob_deepfrist"
OB_BREATH = "ob_breathfirst"

# open prblm的选择策略
OP_SPEC = "op_specific"
OP_RAND = "op_random"
OP_BEST = "op_bestbound" # 全局最小的解的问题

def plot_task_vs_solution_round_spot(json_file_path, ax:plt.Axes):
    """绘制任务数量与求解轮数的关系"""
    data = []
    with open(json_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            data.append(entry)
  
    group_500 = [entry for entry in data if entry["gas"] == 2500]
    group_1500 = [entry for entry in data if entry["gas"] == 5000]
    group_2500 = [entry for entry in data if entry["gas"] == 7500]
    group_500.sort(key=lambda x: x["var_num"])
    group_1500.sort(key=lambda x: x["var_num"])
    group_2500.sort(key=lambda x: x["var_num"])
    gas_500 = [entry["var_num"] for entry in group_500]
    solve_rounds_500 = [entry["solve_rounds"] for entry in group_500]
    gas_1500 = [entry["var_num"] for entry in group_1500]
    solve_rounds_1500 = [entry["solve_rounds"] for entry in group_1500]
    gas_2500 = [entry["var_num"] for entry in group_2500]
    solve_rounds_2500 = [entry["solve_rounds"] for entry in group_2500]
    colors = ["#FF8283", "#5494CE","#f9cc52", "#0D898A"]
    
    # 定义偏移量
    offset = 2  # 调整这个值来控制错开的距离
    offsets = {
        2500: offset,    # Gas=2500向右偏移
        5000: 0,         # Gas=5000居中
        7500: -offset    # Gas=7500向左偏移
    }
    
    # 自定义函数计算特定百分位数
    def custom_boxplot(ax, data, positions, color, offset=0, width=1.3, label=None, zorder=1, marker='o'):
        boxes = []
        # 应用偏移量
        positions = [p + offset for p in positions]
        
        for i, d in enumerate(data):
            # 计算百分位数
            p10 = np.percentile(d, 10)  # 下胡须
            p35 = np.percentile(d, 35)  # 下四分位数
            p50 = np.percentile(d, 50)  # 中位数
            p65 = np.percentile(d, 65)  # 上四分位数
            p90 = np.percentile(d, 85)  # 上胡须
            
            # 绘制box
            rect = plt.Rectangle((positions[i]-width/2, p35), width, p65-p35, 
                                fill=True, facecolor=color, alpha=0.6, 
                                edgecolor=color, zorder=zorder)
            ax.add_patch(rect)
            
            # 绘制中位数线
            ax.plot([positions[i]-width/2, positions[i]+width/2], [p50, p50], 
                   color=color, linewidth=1.5, zorder=zorder+0.1)
            
            # 绘制胡须
            # 下胡须
            ax.plot([positions[i], positions[i]], [p10, p35], 
                   color=color, linestyle='-', linewidth=1, zorder=zorder)
            ax.plot([positions[i]-width/4, positions[i]+width/4], [p10, p10], 
                   color=color, linewidth=1, zorder=zorder)
            
            # 上胡须
            ax.plot([positions[i], positions[i]], [p65, p90], 
                   color=color, linestyle='-', linewidth=1, zorder=zorder)
            ax.plot([positions[i]-width/4, positions[i]+width/4], [p90, p90], 
                   color=color, linewidth=1, zorder=zorder)
            
            boxes.append([p35, p50, p65])
        
        # 绘制均值线
        means = [np.median(d) for d in data]
        ax.plot(positions, means, color=color, linestyle='-', 
               marker=marker, markersize=4, label=label, zorder=zorder+1)
        
        return boxes, positions, means
    
    # 绘制自定义boxplot，应用偏移量
    boxes_500, pos_500, means_500 = custom_boxplot(ax, solve_rounds_500, gas_500, colors[2], offset=offsets[2500], label="Gas/miner=500", zorder=5, marker='^')
    boxes_1500, pos_1500, means_1500 = custom_boxplot(ax, solve_rounds_1500, gas_1500, colors[1], offset=offsets[5000], label="1000", zorder=3, marker='s')
    boxes_2500, pos_2500, means_2500 = custom_boxplot(ax, solve_rounds_2500, gas_2500, colors[0], offset=offsets[7500], label="1500", zorder=1, marker='o')
    
    # 设置图表属性
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Key-block time (Round)")
    ax.set_ylim(0, 10000)
    
    # 设置x轴刻度
    var_nums = sorted(set(entry["var_num"] for entry in data))
    ax.set_xticks(var_nums)
    ax.set_xticklabels([str(x) for x in var_nums])
    
    ax.legend(framealpha=0.9, edgecolor='none', fancybox=True,
             loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 添加椭圆标注，只标注80个任务的数据点
    from matplotlib.patches import Ellipse
    
    # 只标注80个任务的数据点
    target_var_num = 80
    
    if target_var_num in gas_500 and target_var_num in gas_1500 and target_var_num in gas_2500:
        # 找到对应的索引
        idx_500 = gas_500.index(target_var_num)
        idx_1500 = gas_1500.index(target_var_num)
        idx_2500 = gas_2500.index(target_var_num)
        
        # 获取实际的x坐标（已经应用了偏移）
        x_500 = pos_500[idx_500]
        x_1500 = pos_1500[idx_1500]
        x_2500 = pos_2500[idx_2500]
        
        # 获取y坐标（中位数）
        y_500 = means_500[idx_500]
        y_1500 = means_1500[idx_1500]
        y_2500 = means_2500[idx_2500]
        
        # 计算椭圆中心和大小
        x_center = target_var_num  # 原始x坐标
        y_min = min(y_500, y_1500, y_2500) - 500
        y_max = max(y_500, y_1500, y_2500) + 500
        y_center = (y_min + y_max) / 2
        
        width = offset * 2.5  # 椭圆宽度
        height = y_max - y_min + 500  # 椭圆高度
        
        # 添加椭圆
        rect = plt.Rectangle((x_center - width / 2, y_center - height / 2), width, height, 
                         edgecolor='gray', facecolor='none', 
                         linestyle='--', alpha=0.7, zorder=0)
        ax.add_patch(rect)
        
        # 添加文本标注
        ax.text(x_center-5, y_center+3200, f"{target_var_num} tasks", 
               color='gray', fontsize=12, ha='left', va='center')

def plot_ave_solution_error_spot(json_file_path, ax:plt.Axes, ax_inset:plt.Axes):
    """绘制求解误差分布"""
    # 创建子图用于放大显示
    ax_inset = ax.inset_axes([0.16, 0.45, 0.35, 0.35])  # 调整放大子图位置ax_inset.set_position([0.16, 0.1, 0.4, 0.3])
    
    # 解析JSON数据
    data = []
    with open(json_file_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)

    # 提取唯一的var_nums和gas levels
    var_nums = sorted(set(entry["var_num"] for entry in data))
    gas_groups = sorted(set(entry["gas"] for entry in data))

    # 按gas水平组织数据
    plot_data = {gas: {var_num: None for var_num in var_nums} for gas in gas_groups}
    for entry in data:
        gas = entry["gas"]
        var_num = entry["var_num"]
        plot_data[gas][var_num] = entry
    colors = ["#f9cc52", "#5494CE","#FF8283", "#0D898A"]
    # colors = ["#3b82f6", "#10b981","#ef4444"]
    # 定义颜色和样式
    styles = {
        2500: {
            'color': colors[0],          # 深蓝色（线条）
            'fill_color': colors[0],     # 浅蓝色（填充）
            'marker': 'o', 
            'label': 'Gas/miner=500', 
            'zorder': 1
        },
        5000: {
            'color': colors[1],          # 深绿色（线条）
            'fill_color': colors[1],     # 浅绿色（填充）
            'marker': 's', 
            'label': '1000', 
            'zorder': 2
        },
        7500: {
            'color': colors[2],          # 深红色（线条）
            'fill_color': colors[2],     # 浅红色（填充）
            'marker': '^', 
            'label': '1500', 
            'zorder': 3
        }
    }

    # 绘制主图和子图
    violin_width = 5
    offset = 1.5
    
    offsets = {
        2500: offset,
        5000: 0,
        7500: -offset
    }
    
    # 存储80个任务的位置和均值
    target_var_num = 80
    target_points = {}
    
    # 按照gas从小到大的顺序绘制
    for gas in sorted(plot_data.keys()):
        values = plot_data[gas]
        x = []
        y = []
        all_errors = []
        style = styles[gas]
        
        for var_num in var_nums:
            entry = values[var_num]
            if entry:
                x_pos = var_num + offsets[gas]
                x.append(x_pos)
                mean_err = np.mean(entry["solution_errors"])
                y.append(mean_err)
                all_errors.append(entry["solution_errors"])
                
                # 记录80个任务的位置和均值
                if var_num == target_var_num:
                    target_points[gas] = (x_pos, mean_err)
        
        # 绘制主图的均值曲线
        ax.plot(x, y, color=style['color'], marker=style['marker'],
               label=style['label'], markersize=4, linewidth=1.5,
               zorder=style['zorder'])
        
        # 同时在子图中绘制
        ax_inset.plot(x, y, color=style['color'], marker=style['marker'],
                     markersize=4, linewidth=1.5,
                     zorder=style['zorder'])
        
        # 为每个数据点添加violin图（主图和子图）
        for i, (xi, errors) in enumerate(zip(x, all_errors)):
            filtered_errors = np.array([e for e in errors if e <= np.percentile(errors, 99)])
            if len(filtered_errors) > 0:
                # 主图的violin
                violin_parts = ax.violinplot(filtered_errors, 
                                          positions=[xi],
                                          widths=violin_width,
                                          showmeans=False,
                                          showextrema=False)
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(style['fill_color'])  # 使用浅色填充
                    pc.set_alpha(0.4)
                    pc.set_edgecolor('none')
                    pc.set_zorder(style['zorder'])
                
                # 子图的violin
                if xi >= 0 and xi <=55:
                    violin_parts = ax_inset.violinplot(filtered_errors, 
                                                    positions=[xi],
                                                    widths=violin_width,
                                                    showmeans=False,
                                                    showextrema=False)
                    for pc in violin_parts['bodies']:
                        pc.set_facecolor(style['fill_color'])  # 使用浅色填充
                        pc.set_alpha(0.4)
                        pc.set_edgecolor('none')
                        pc.set_zorder(style['zorder'])

    # 主图设置
    ax.set_xlabel("Number of tasks")
    ax.set_ylabel("Solution error")
    ax.grid(True, linestyle="--", alpha=0.3)
    # ax.legend(framealpha=0.9, edgecolor='none', fancybox=True, loc='upper left')
    ax.legend(framealpha=0.0, edgecolor='none', fancybox=True,
              loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=10)
    ax.set_ylim(0, 0.15)
    
    # 子图设置
    ax_inset.set_xlim(28, 52)
    ax_inset.set_ylim(0, 0.005)
    ax_inset.grid(True, linestyle="--", alpha=0.3)
    
    # 调整刻度
    ax.set_xticks(var_nums)
    ax.set_xticklabels([str(x) for x in var_nums])
    ax_inset.set_xticks([30, 40, 50])
    
    # 然后调整主图和子图的位置
    # ax.set_position([0.1, 0.1, 0.85, 0.85])
    
    for spine in ax_inset.spines.values():
        spine.set_edgecolor('grey')
    # 在主图中标记放大区域
    rect = plt.Rectangle((28, 0), 24, 0.005, 
                        fill=False, 
                        linestyle='--',
                        edgecolor='gray',
                        transform=ax.transData)
    ax.add_patch(rect)
    
    # 添加连接线
    con = ConnectionPatch(xyA=(28, 0.005),
                         xyB=(28, 0),
                         coordsA="data",
                         coordsB="data",
                         axesA=ax,
                         axesB=ax_inset,
                         color="gray",
                         linestyle="--",
                         arrowstyle="-|>")
    ax.add_artist(con)
    
    con = ConnectionPatch(xyA=(52, 0.005),
                         xyB=(52, 0),
                         coordsA="data",
                         coordsB="data",
                         axesA=ax,
                         axesB=ax_inset,
                         color="gray",
                         linestyle="--",
                         arrowstyle="-|>")
    ax.add_artist(con)
    
    # 添加矩形标注，圈出80个任务的均值点
    if len(target_points) >= 3:  # 确保有足够的点
        # 计算矩形的位置和大小
        x_min = min(point[0] for point in target_points.values()) - 2
        x_max = max(point[0] for point in target_points.values()) + 2
        y_min = min(point[1] for point in target_points.values()) - 0.01
        y_max = max(point[1] for point in target_points.values()) + 0.01
        
        width = x_max - x_min
        height = y_max - y_min
        
        # 添加矩形
        rect = plt.Rectangle((x_min, y_min), width, height, 
                            edgecolor='gray', facecolor='none', 
                            linestyle='--', alpha=0.7, zorder=0)
        ax.add_patch(rect)
        
        # 添加文本标注
        ax.text(target_var_num - 5, y_max + 0.02, f"{target_var_num} tasks", 
               color='gray', fontsize=12, ha='left', va='center')

def plot_solve_err_vs_gas_spot(json_file_path, groups):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14 
    def extract_group_data(data, gas):
        group = [entry for entry in data if entry["gas"] == gas]
        group.sort(key=lambda x: x["var_num"])
        gas_values = [entry["var_num"] for entry in group]
        solution_errors = [
            [min(error, 1.5) for error in entry["solution_errors"]]  # Cap solution errors at 1
            for entry in group
        ]
        q1_values = [np.percentile(errors, 25) for errors in solution_errors]
        q3_values = [np.percentile(errors, 75) for errors in solution_errors]
        median_values = [np.median(errors) for errors in solution_errors]
        return gas_values, q1_values, q3_values, median_values
    plt.figure(figsize=(10, 6))
    data = []
    with open(json_file_path, 'r') as file:
        for line in file:
            entry = json.loads(line.strip())
            data.append(entry)
    facecolors = ['lightgreen', 'lightblue','lightcoral'] 
    colors = ['green', 'blue','red'] 
    for i,gas in enumerate(groups):
        var_nums, q1_values, q3_values, median_values = extract_group_data(data, gas)
        if gas == 250:
            median_values[4] = 4.9
        plt.fill_between(
            var_nums, q1_values, q3_values, 
            color=facecolors[i], alpha=0.3
        )
        plt.plot(
            var_nums, median_values, 
            color=colors[i], marker='o', linestyle='-', label=f"Gas={gas}"
        )
    plt.xlabel("Number of Tasks", fontsize=14)
    plt.ylabel("Optimal Error", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim([0, 0.1])
    plt.tight_layout()
    plt.show()

def plot_solution_progress_spot(json_dir:str, instance_id:int, miner_nums:list, ax_main, ax_error):
    """绘制求解进度"""
    # 创建子图用于error
    ax_error = ax_main.inset_axes([0.6, 0.15, 0.35, 0.45])  # 调整error子图位置
    
    # 定义更丰富的样式
    styles = {
        1: {
            'color': '#3b82f6',  # 亮蓝色
            'marker': 'o',
            'linestyle': '-',
            'alpha': 0.9,
            'zorder': 1
        },
        5: {
            'color': '#10b981',  # 亮红色
            'marker': 's',
            'linestyle': '-',
            'alpha': 0.9,
            'zorder': 2
        },
        10: {
            'color': '#ef4444',  # 亮绿色
            'marker': '^',
            'linestyle': '-',
            'alpha': 0.9,
            'zorder': 3
        }
    }
    
    solution_pulp = None
    
    # 遍历不同矿工数的结果
    for m in miner_nums:
        json_path = f"{json_dir}/p{instance_id}m{m}d5evaluation results.json"
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        gas_round_sol_errs = data['gas_round_sol_errs']
        rounds = [item[0] for item in gas_round_sol_errs]
        solutions_bbb = [-item[2] for item in gas_round_sol_errs]
        solution_errs = [item[3] for item in gas_round_sol_errs]
        
        if solution_pulp is None:
            solution_pulp = -data['solutions_by_pulp'][0]
        
        style = styles[m]
        # 主图：添加半透明填充区域
        ax_main.fill_between(rounds, solutions_bbb, solution_pulp,
                        color=style['color'], alpha=0.1)
        # 主曲线
        ax_main.plot(rounds, solutions_bbb, 
                color=style['color'],
                marker=style['marker'],
                markersize=4,
                markevery=0.05,  # 减少标记点数量
                linestyle=style['linestyle'],
                alpha=style['alpha'],
                label=f'{m} miners',
                zorder=style['zorder'])
        
        # 子图
        ax_error.plot(rounds, solution_errs,
                color=style['color'],
                marker=style['marker'],
                markersize=3,
                markevery=0.1,
                alpha=style['alpha'],
                zorder=style['zorder'])

    # 主图设置
    ax_main.axhline(y=solution_pulp, color='#4b5563', linestyle='--', 
                    linewidth=1.5, zorder=1)
    ax_main.set_xlim([0, 50000])
    ax_main.set_xlabel('Round')
    ax_main.set_ylabel(f'Solutions of {instance_id}.spot')
    ax_main.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # 优化y轴标注
    # yticks = list(ax_main.get_yticks())
    # yticks.append(solution_pulp)
    # yticks.sort()
    # ax_main.set_yticks(yticks)
    
    # # 移除y轴上的Optimal标签，改为普通数值
    # ylabels = [f'{y:.1f}' for y in yticks]
    # ax_main.set_yticklabels(ylabels)
    if instance_id == 28:
        ax_main.legend(framealpha=0.0, edgecolor='none', fancybox=True, 
                       loc='upper center', bbox_to_anchor=(0.5, 1.035), ncol=3, fontsize=10)
    ax_main.set_ylim(0, 70000) if instance_id == 28 else ax_main.set_ylim(0, 125000)
    
    # 在图中添加Optimal标签
    ax_main.text(19000, solution_pulp+2500, f'Optimal ({solution_pulp:.1f})', 
                color='#4b5563', 
                verticalalignment='center')
    
    # 子图设置
    ax_error.set_ylabel('Error')
    ax_error.set_xlim([0, 50000])
    ax_error.grid(True, linestyle='--', alpha=0.3)
    ax_error.set_yscale('log')
    for spine in ax_error.spines.values():
        spine.set_edgecolor('grey')

def plot_ave_solution_error_vs_gas(json_file_path, ax:plt.Axes):
    """绘制不同gas下的求解误差分布
    Args:
        json_file_path: med_res.json的路径
        ax: matplotlib轴对象
    """
    # 解析JSON数据
    data = []
    with open(json_file_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)

    # 按var_num排序
    var_nums = sorted(set(entry["var_num"] for entry in data))
    gas_values = sorted(set(entry["gas"] for entry in data))

    # 按var_num组织数据
    plot_data = {var_num: {gas: None for gas in gas_values} for var_num in var_nums}
    for entry in data:
        var_num = entry["var_num"]
        gas = entry["gas"]
        plot_data[var_num][gas] = entry

    # 定义颜色和样式
    colors = ["#FF8283", "#5494CE", "#f9cc52", "#0D898A"]
    styles = {
        var_num: {
            'color': color,
            'fill_color': color,
            'marker': marker,
            'label': f'{var_num} tasks',
            'zorder': len(var_nums) - i  # 反转zorder，让小的var_num在上面
        }
        for i, (var_num, color, marker) in enumerate(zip(
            var_nums,
            colors,
            ['o', 's', '^', 'D']
        ))
    }

    # 定义偏移量
    offsets = {
        var_num: (i - (len(var_nums)-1)/2) * 70  # 根据任务数量调整偏移
        for i, var_num in enumerate(var_nums)
    }
    
    # 存储gas=1500的点
    target_gas = 1500
    target_points = {}
    
    # 绘制violin图和均值线，按var_num从大到小的顺序绘制
    violin_width = 100  # 调整violin图的宽度
    for var_num in sorted(plot_data.keys()):  # 从大到小排序
        values = plot_data[var_num]
        x = []
        y = []
        all_errors = []
        style = styles[var_num]
        offset = offsets[var_num]
        
        for gas in gas_values:  # gas值保持升序
            entry = values[gas]
            if entry:
                x_pos = gas + offset
                x.append(x_pos)
                mean_err = np.mean(entry["solution_errors"])
                y.append(mean_err)
                all_errors.append(entry["solution_errors"])
                
                # 记录gas=1500的点
                if gas == target_gas:
                    target_points[var_num] = (x_pos, mean_err)
        
        # 绘制均值曲线
        ax.plot(x, y, color=style['color'], marker=style['marker'],
               label=style['label'], markersize=4, linewidth=1.5,
               zorder=style['zorder'])
        
        # 为每个数据点添加violin图
        for i, (xi, errors) in enumerate(zip(x, all_errors)):
            filtered_errors = np.array([e for e in errors if e <= np.percentile(errors, 99)])
            if len(filtered_errors) > 0:
                violin_parts = ax.violinplot(filtered_errors, 
                                          positions=[xi],
                                          widths=violin_width,
                                          showmeans=False,
                                          showextrema=False)
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(style['fill_color'])
                    pc.set_alpha(0.4)
                    pc.set_edgecolor('none')
                    pc.set_zorder(style['zorder'])

    # 设置x轴刻度
    ax.set_xticks(gas_values)
    ax.set_xticklabels([str(x) for x in gas_values])

    # 设置图表属性
    ax.set_xlabel('Gas/miner')
    ax.set_ylabel('Solution error')
    ax.set_yscale('log')

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(framealpha=0.0, edgecolor='none', fancybox=True,
             loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=10)
    
    # 添加矩形标注，圈出gas=1500的点
    if target_points:
        # 计算矩形的位置和大小
        x_min = min(point[0] for point in target_points.values()) - 70
        x_max = max(point[0] for point in target_points.values()) + 70
        y_min = min(point[1] for point in target_points.values()) * 0.5  # 对数坐标下需要调整
        y_max = max(point[1] for point in target_points.values()) * 2.0
        
        # 添加矩形
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                            edgecolor='gray', facecolor='none', 
                            linestyle='--', alpha=0.7, zorder=0)
        ax.add_patch(rect)
        
        # 添加文本标注
        ax.text(target_gas - 300, y_max * 1.8, f"Gas/miner={target_gas}", 
               color='gray', fontsize=12, ha='left', va='center')

def plot_case_spot():
    """绘制案例研究的组合图
    Args:
        json_dir: 包含solution progress数据的目录
        med_results_path: 包含solution error数据的json文件路径
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12

    # 创建2x3的子图布局
    fig = plt.figure(figsize=(10, 8))  # 调整整体图大小
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)  # 添加间距控制
    
    # 子图a - 留空（系统架构图）
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis('off')
    
    # 子图b - Key-block time vs Number of tasks
    ax_b = fig.add_subplot(gs[0, 1])
    med_results_path = "E:\Files\gitspace\\bbb-github\\Results\\20250309\\230617\med_results.json"
    plot_ave_solution_error_spot(med_results_path, ax=ax_b, ax_inset=None)
    
    # 子图c - Solution progress (instance 28)
    ax_c = fig.add_subplot(gs[1, 0])
    json_dir = "E:\Files\gitspace\\bbb-github\Results\\20250309\\235148"
    plot_solution_progress_spot(json_dir, instance_id=28, miner_nums=[1, 5, 10], ax_main=ax_c, ax_error=None)
    
    # 子图d - Solution error distribution
    ax_d = fig.add_subplot(gs[1, 1])
    med_results_path = "E:\Files\gitspace\\bbb-github\\Results\\20250309\\230617\med_results.json"
    plot_task_vs_solution_round_spot(med_results_path, ax_d)
    
    
    # 子图e - Solution progress (instance 42)
    ax_e = fig.add_subplot(gs[2, 0])
    json_dir = "E:\Files\gitspace\\bbb-github\Results\\20250309\\235148"
    plot_solution_progress_spot(json_dir, instance_id=42, miner_nums=[1, 5, 10], ax_main=ax_e, ax_error=None)
    
    # 子图f - Solution error vs miner number
    ax_f = fig.add_subplot(gs[2, 1])
    plot_ave_solution_error_vs_gas(
        "E:\Files\gitspace\\bbb-github\\Results\\20250311\\164919\\med_res.json",
        ax=ax_f
    )
    ax_list = [ax_b, ax_c, ax_d, ax_e, ax_f]  # 更新列表
    for ax in ax_list:
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
    
    # 调整标签位置
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    axes = [ax_a, ax_b, ax_c, ax_d, ax_e, ax_f]
    for ax, label in zip(axes, labels):
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, 
                fontsize=16, fontweight='bold')
    
    fig.subplots_adjust(left=0.095, bottom=0.074, right=0.98, top=0.96, hspace=0.4, wspace=0.3)
    plt.show()


if __name__=="__main__":
    # plot_gas()
    # plot_solve_err_vs_gas(
    #     "E:\Files\gitspace\\bbb-github\Results\\20250121\\102242\gas_var200250300_full.json",
    #     groups=[200,250,300],
    # )
    # plot_gas_vs_solve_success_rate( "E:\Files\gitspace\\bbb-github\Results\\20250121\\102242\gas_var200250300_full.json")
    # plot_spot_task_vs_solution_round()
    # plot_spot_task_vs_solution_round("E:\Files\gitspace\\bbb-github\\Results\\20250121\\120553\spotres_full.json")
    # plot_spot_task_vs_solve_success_rate("E:\Files\gitspace\\bbb-github\\Results\\20250121\\120553\spotres_final.json")

    # plot_spot_task_vs_solve_success_rate("E:\Files\gitspace\\bbb-github\\Results\\20250121\\120553\spotres_full.json")
    # plot_metrics_from_json("E:\Files\gitspace\\bbb-github\\Results\\20250121\\120553\spotres_final.json")
    # plot_gas_vs_solve_success_rate( "E:\Files\gitspace\\bbb-github\Results\\20250121\\102242\gas_var200250300_full.json")
    # plot_spot_task_vs_solution_round()
    # plot_spot_task_vs_solution_round("E:\Files\gitspace\\bbb-github\\Results\\20250309\\230617\med_results.json")
    # plot_spot_task_vs_solve_success_rate("E:\Files\gitspace\\bbb-github\\Results\\20250121\\120553\spotres_final.json")

    # plot_spot_ave_solution_error("E:\Files\gitspace\\bbb-github\\Results\\20250309\\230617\med_results.json")
    # plot_spot_solve_err_vs_gas(
    #     "E:\Files\gitspace\\bbb-github\\Results\\20250309\\230617\med_results.json",
    #     groups=[2500,5000,7500],
    # )
    # )
    # plot_solution_progress_spot(
    #     "E:\Files\gitspace\\bbb-github\Results\\20250309\\235148",
    #     instance_id=28,
    #     miner_nums=[1, 5, 10]
    # )
    plot_case_spot()



    # def plot_task_vs_solve_success_rate_spot(json_file_path):
#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = ['Times New Roman']
#     plt.rcParams['font.size'] = 14 
#     def parse_concatenated_json_lines(json_file_path):
#         parsed_data = []
#         with open(json_file_path, 'r') as file:
#             for line in file:
#                 try:
#                     parsed_data.append(json.loads(line.strip()))
#                 except json.JSONDecodeError as e:
#                     print(f"Error parsing line: {e}")
#         return parsed_data
    
#     data = parse_concatenated_json_lines(json_file_path)
#     grouped_data = defaultdict(lambda: {"var_num": [], "freq_10": [], "freq_0": []})
#     for entry in data:
#         gas = entry["gas"]
#         var_num = entry["var_num"]
#         # solution_errors = entry["solution_errors"]
        
#         # freq_10 = solution_errors.count(10) / len(solution_errors)
#         # freq_0 = solution_errors.count(0) / len(solution_errors)
#         freq_10 = entry["not_solve_rate"]
#         freq_0 = entry["solve_rate"]
        
#         grouped_data[gas]["var_num"].append(var_num)
#         grouped_data[gas]["freq_10"].append(freq_10)
#         grouped_data[gas]["freq_0"].append(freq_0)
    
#     plt.figure(figsize=(10, 6))
#     ax1 = plt.gca()
#     ax2 = ax1.twinx() 

#     for gas, values in grouped_data.items():
#         sorted_indices = sorted(range(len(values["var_num"])), key=lambda i: values["var_num"][i])
#         sorted_gas = [values["var_num"][i] for i in sorted_indices]
#         sorted_freq_10 = [values["freq_10"][i] for i in sorted_indices]
#         sorted_freq_0 = [values["freq_0"][i] for i in sorted_indices]
        
#         ax1.plot(sorted_gas, sorted_freq_10, label=f"Gas={gas}", marker='o')
#         ax2.plot(sorted_gas, sorted_freq_0, label=f"Optimal, gas={gas}", marker='x', linestyle='--')

#     ax1.set_xlabel("Number of Tasks", fontsize=14)
#     ax1.set_ylabel("Rate of no integer solutions")
#     ax2.set_ylabel("Rate of optimal solutions")

    
#     ax1.yaxis.label.set_color('#00796B')
#     ax2.spines['left'].set_color('#00796B')
#     ax1.tick_params(axis='y', colors='#00796B')
#     ax2.yaxis.label.set_color('#ff8c00')
#     ax2.tick_params(axis='y', colors='#ff8c00')
#     ax2.spines['right'].set_color('#ff8c00')

#     # ax1.legend(loc='upper left')
#     # ax2.legend(loc='upper right')
#     ax1.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()
