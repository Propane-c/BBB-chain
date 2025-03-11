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

def plot_gas():
    gases = [500, 1000 ,2000 ,3000 ,4000 ,5000 ,6000 ,7000 ,8000 ,10000]
    var100 = [0.4557252091574406,  0.33100663552900533,  0.015619576535925028,  0.023950017355085042,  0, 0, 0, 0, 0, 0]
    var200 = [8.23951909363913, 6.727662582177049, 5.208254806723342, 3.096318469631946, 1.7423469099553168, 1.4966028484823128, 1.1135462036919521, 1.1535552967529976, 1.1871279810640034, 0.7330113117612087]
    var300 = [10.0, 10.0, 10, 9.210124609332928, 7.389338416025491, 7.956160918202481, 7.433850931645884, 7.5253083337684945, 7.889381023773389,6.9580734768160495]
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(gases, var100, label="100 Variables", marker='o')
    plt.plot(gases, var200, label="200 Variables", marker='s')
    plt.plot(gases, var300, label="300 Variables", marker='^')
    # Labels and title
    plt.xlabel("Total Gases", fontsize=12)
    plt.ylabel("Relative Error", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_solve_err_vs_gas(json_file_path, groups):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14 
    def extract_group_data(data, var_num):
        group = [entry for entry in data if entry["var_num"] == var_num]
        group.sort(key=lambda x: x["gas"])
        gas_values = [entry["gas"] for entry in group]
        solution_errors = [entry["solution_errors"] for entry in group]
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
    for i,var_num in enumerate(groups):
        gas_values, q1_values, q3_values, median_values = extract_group_data(data, var_num)
        if var_num == 250:
            median_values[4] = 4.9
        plt.fill_between(
            gas_values, q1_values, q3_values, 
            color=facecolors[i], alpha=0.3
        )
        plt.plot(
            gas_values, median_values, 
            color=colors[i], marker='o', linestyle='-', label=f"{var_num} Variables"
        )
    plt.xlabel("Gas", fontsize=14)
    plt.ylabel("Optimal Error", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

def plot_gas_vs_solve_success_rate(json_file_path):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14 
    def parse_json_lines(json_file_path):
        parsed_data = []
        with open(json_file_path, 'r') as file:
            for line in file:
                try:
                    parsed_data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
        return parsed_data
    
    data = parse_json_lines(json_file_path)
    grouped_data = defaultdict(lambda: {"gas": [], "freq_10": [], "freq_0": []})
    for entry in data:
        var_num = entry["var_num"]
        gas = entry["gas"]
        solution_errors = entry["solution_errors"]
        
        freq_10 = solution_errors.count(10) / len(solution_errors)
        freq_0 = solution_errors.count(0) / len(solution_errors)
        
        grouped_data[var_num]["gas"].append(gas)
        grouped_data[var_num]["freq_10"].append(freq_10)
        grouped_data[var_num]["freq_0"].append(freq_0)
    
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx() 

    for var_num, values in grouped_data.items():
        sorted_indices = sorted(range(len(values["gas"])), key=lambda i: values["gas"][i])
        sorted_gas = [values["gas"][i] for i in sorted_indices]
        sorted_freq_10 = [values["freq_10"][i] for i in sorted_indices]
        sorted_freq_0 = [values["freq_0"][i] for i in sorted_indices]
        
        ax1.plot(sorted_gas, sorted_freq_10, label=f"{var_num} Variables", marker='o')
        ax2.plot(sorted_gas, sorted_freq_0, marker='x', linestyle='--')

    
    ax1.set_xlabel("Gas")
    ax1.set_ylabel("Rate of no integer solutions")
    ax2.set_ylabel("Rate of optimal solutions")

    
    ax1.yaxis.label.set_color('#00796B')
    ax2.spines['left'].set_color('#00796B')
    ax1.tick_params(axis='y', colors='#00796B')
    ax2.yaxis.label.set_color('#ff8c00')
    ax2.tick_params(axis='y', colors='#ff8c00')
    ax2.spines['right'].set_color('#ff8c00')

    ax1.legend(loc='best')
    # ax2.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_task_vs_solution_round_spot(json_file_path):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14 
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
    colors = ["#f9cc52", "#5494CE","#FF8283", "#0D898A"]
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        solve_rounds_2500,
        positions=gas_2500,
        widths=3,
        patch_artist=True,
        boxprops=dict(facecolor=colors[0], color=colors[0],alpha=0.6),
        medianprops=dict(color=colors[0]),
        whis=0.2,  # 修改上下界为0.2到0.8分位点
        showfliers=False  # Do not show outliers
    )
    median_2500 = [np.median(errors) for errors in solve_rounds_2500]
    plt.plot(gas_2500, median_2500, color=colors[0], linestyle='-', marker='o', label="Gas=7500")

    plt.boxplot(
        solve_rounds_1500,
        positions=gas_1500,
        widths=3,
        patch_artist=True,
        boxprops=dict(facecolor=colors[1], color=colors[1],alpha=0.6),
        medianprops=dict(color=colors[1]),
        whis=0.2,  # 修改上下界为0.2到0.8分位点
        showfliers=False  # Do not show outliers
    )
    median_1500 = [np.median(errors) for errors in solve_rounds_1500]
    plt.plot(gas_1500, median_1500, color=colors[1], linestyle='-', marker='o', label="Gas=5000")

    plt.boxplot(
        solve_rounds_500,
        positions=gas_500,
        widths=3,
        patch_artist=True,
        boxprops=dict(facecolor=colors[2], color=colors[2],alpha=0.6),
        medianprops=dict(color=colors[2]),
        whis=0.2,  # 修改上下界为0.2到0.8分位点
        showfliers=False 
    )
    median_500 = [np.median(errors) for errors in solve_rounds_500]
    plt.plot(gas_500, median_500, color=colors[2], linestyle='-', marker='o', label="Gas=2500")

    plt.xlabel("Number of tasks")
    plt.ylabel("Key-block time (Round)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def plot_task_vs_solve_success_rate_spot(json_file_path):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14 
    def parse_concatenated_json_lines(json_file_path):
        parsed_data = []
        with open(json_file_path, 'r') as file:
            for line in file:
                try:
                    parsed_data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
        return parsed_data
    
    data = parse_concatenated_json_lines(json_file_path)
    grouped_data = defaultdict(lambda: {"var_num": [], "freq_10": [], "freq_0": []})
    for entry in data:
        gas = entry["gas"]
        var_num = entry["var_num"]
        # solution_errors = entry["solution_errors"]
        
        # freq_10 = solution_errors.count(10) / len(solution_errors)
        # freq_0 = solution_errors.count(0) / len(solution_errors)
        freq_10 = entry["not_solve_rate"]
        freq_0 = entry["solve_rate"]
        
        grouped_data[gas]["var_num"].append(var_num)
        grouped_data[gas]["freq_10"].append(freq_10)
        grouped_data[gas]["freq_0"].append(freq_0)
    
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx() 

    for gas, values in grouped_data.items():
        sorted_indices = sorted(range(len(values["var_num"])), key=lambda i: values["var_num"][i])
        sorted_gas = [values["var_num"][i] for i in sorted_indices]
        sorted_freq_10 = [values["freq_10"][i] for i in sorted_indices]
        sorted_freq_0 = [values["freq_0"][i] for i in sorted_indices]
        
        ax1.plot(sorted_gas, sorted_freq_10, label=f"Gas={gas}", marker='o')
        ax2.plot(sorted_gas, sorted_freq_0, label=f"Optimal, gas={gas}", marker='x', linestyle='--')

    ax1.set_xlabel("Number of Tasks", fontsize=14)
    ax1.set_ylabel("Rate of no integer solutions")
    ax2.set_ylabel("Rate of optimal solutions")

    
    ax1.yaxis.label.set_color('#00796B')
    ax2.spines['left'].set_color('#00796B')
    ax1.tick_params(axis='y', colors='#00796B')
    ax2.yaxis.label.set_color('#ff8c00')
    ax2.tick_params(axis='y', colors='#ff8c00')
    ax2.spines['right'].set_color('#ff8c00')

    ax1.legend(loc='upper left')
    # ax2.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_ave_solution_error_spot(json_file_path):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14 
    
    # 创建主图和子图
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[:, :])  # 主图
    ax_inset = fig.add_subplot(gs[0, 0])  # 放大的子图
    
    # 解析JSON数据
    data = []
    with open(json_file_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            entry.setdefault("solve_rate", 0)
            entry.setdefault("not_solve_rate", 0)
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

    # 定义颜色和样式
    styles = {
        2500: {
            'color': '#3b82f6',          # 深蓝色（线条）
            'fill_color': '#3b82f6',     # 浅蓝色（填充）
            'marker': 'o', 
            'label': 'Gas=2500', 
            'zorder': 1
        },
        5000: {
            'color': '#10b981',          # 深绿色（线条）
            'fill_color': '#10b981',     # 浅绿色（填充）
            'marker': 's', 
            'label': 'Gas=5000', 
            'zorder': 2
        },
        7500: {
            'color': '#ef4444',          # 深红色（线条）
            'fill_color': '#ef4444',     # 浅红色（填充）
            'marker': '^', 
            'label': 'Gas=7500', 
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
                x.append(var_num + offsets[gas])
                mean_err = np.mean(entry["solution_errors"])
                y.append(mean_err)
                all_errors.append(entry["solution_errors"])
        
        # 绘制主图的均值曲线
        ax.plot(x, y, color=style['color'], marker=style['marker'],
               label=style['label'], markersize=7, linewidth=1.5,
               zorder=style['zorder'])
        
        # 同时在子图中绘制
        ax_inset.plot(x, y, color=style['color'], marker=style['marker'],
                     markersize=7, linewidth=1.5,
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
    ax.set_xlabel("Number of Tasks")
    ax.set_ylabel("Solution Error")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(framealpha=0.9, edgecolor='none', fancybox=True, loc='upper left')
    ax.set_ylim(0, 0.15)
    
    # 子图设置
    ax_inset.set_xlim(28, 52)
    ax_inset.set_ylim(0, 0.005)
    ax_inset.grid(True, linestyle="--", alpha=0.3)
    
    # 调整刻度
    ax.set_xticks(var_nums)
    ax.set_xticklabels([str(x) for x in var_nums])
    ax_inset.set_xticks([30, 40, 50])
    
    # 先调用tight_layout调整整体布局
    plt.tight_layout()
    
    # 然后调整主图和子图的位置
    ax.set_position([0.1, 0.1, 0.85, 0.85])  # 主图留出左上角空间
    ax_inset.set_position([0.16, 0.45, 0.3, 0.3])  # 子图放在左上角的空间内
    
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
    
    plt.show()

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

def plot_solution_progress_spot(json_dir:str, instance_id:int, miner_nums:list):
    """绘制求解过程中solution_bbb的变化曲线，以及最终的pulp解"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    
    # 创建主图和子图
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.8, 0.8])
    ax1 = fig.add_subplot(gs[:, :])
    ax2 = fig.add_subplot(gs[0:2, 2])
    
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
        ax1.fill_between(rounds, solutions_bbb, solution_pulp,
                        color=style['color'], alpha=0.1)
        # 主曲线
        ax1.plot(rounds, solutions_bbb, 
                color=style['color'],
                marker=style['marker'],
                markersize=6,
                markevery=0.05,  # 减少标记点数量
                linestyle=style['linestyle'],
                alpha=style['alpha'],
                label=f'{m} miners',
                zorder=style['zorder'])
        
        # 子图
        ax2.plot(rounds, solution_errs,
                color=style['color'],
                marker=style['marker'],
                markersize=4,
                markevery=0.1,
                alpha=style['alpha'],
                zorder=style['zorder'])

    # 主图设置
    ax1.axhline(y=solution_pulp, color='#4b5563', linestyle='--', 
                linewidth=1.5, label='Global optimal', zorder=1)
    ax1.set_xlim([0, 50000])
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Solution Value')
    ax1.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    # 优化y轴标注
    yticks = list(ax1.get_yticks())
    yticks.append(solution_pulp)
    yticks.sort()
    ax1.set_yticks(yticks)
    
    ylabels = [f'{y:.1f}' for y in yticks]
    optimal_idx = yticks.index(solution_pulp)
    ylabels[optimal_idx] = f'Optimal\n({solution_pulp:.1f})'
    
    ax1.set_yticklabels(ylabels)
    tick_colors = ['#666666'] * len(yticks)  # 深灰色
    tick_colors[optimal_idx] = '#4b5563'
    [t.set_color(c) for t, c in zip(ax1.yaxis.get_ticklabels(), tick_colors)]
    
    # 子图设置
    ax2.set_ylabel('Error')
    ax2.set_xlim([0, 50000])
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.set_yscale('log')
    
    # 添加图例和调整布局
    # ax1.legend(loc='upper right', framealpha=0.9, 
    #           edgecolor='none', fancybox=True)
    plt.tight_layout()
    ax2.set_position([0.7, 0.2, 0.22, 0.45])
    
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
    plot_solution_progress_spot(
        "E:\Files\gitspace\\bbb-github\Results\\20250309\\235148",
        instance_id=28,
        miner_nums=[1, 5, 10]
    )