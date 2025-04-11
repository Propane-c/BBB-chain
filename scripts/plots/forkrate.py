import json
import math
import pathlib
import sys
import time
from collections import defaultdict
from itertools import groupby

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
    Patch,
    PathPatch,
    Rectangle,
)
from matplotlib.path import Path
from matplotlib.ticker import PercentFormatter
from scipy.signal import find_peaks, peak_prominences
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde

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


def plot_fork_rate():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times new roman']
    plt.rcParams['font.size'] = 20
    # file_path = pathlib.Path.cwd() / "Result_Data\\fig4_20250329\\med_results.json"
    file_path = pathlib.Path.cwd() / "Result_Data\\1226v100_50m1_20.json"
    data_list = []
    with open(file_path, 'r') as f:
        jsondata_list = f.read().split('\n')[:-1]
        for jsondata in jsondata_list:
            data_list.append(json.loads(jsondata))
        df = pd.DataFrame(data_list)
    df = df.sort_values(by=["var_num", "difficulty", "miner_num"])
    df['main'] = df['ave_acp_subpair_num']
    df['fork'] = df['ave_subpair_num'] - df['ave_acp_subpair_num']
    df['unpub'] = df['ave_subpair_unpubs'] - df['ave_subpair_num']

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # 只选择 var_num 为 100 和 150 的数据
    var_nums = [50, 100]
    colors = ["#5494CE","#FF8283", "#0D898A", "#f9cc52",
              "#BEA9E9", "#00B0F0", "#66A266","#F2A663", 
              "#32CD32", "#FFF700",  "#0096FF", ]
    linestyles = ['--', '-']  # 虚线和实线
    markers = ['x', 'o', 's', 'D', '+', 'P','^','v', '*', 'H', '<']
    
    for i, var_num in enumerate(var_nums):
        df_filtered = df[df['var_num'] == var_num]
        difficulties = df_filtered['difficulty'].unique()
        
        for j, difficulty in enumerate(sorted(difficulties)):
            df_subset = df_filtered[df_filtered['difficulty'] == difficulty]
            grouped = df_subset.groupby('miner_num')['ave_mb_forkrate'].mean().reset_index()
            
            ax.plot(grouped['miner_num'], grouped['ave_mb_forkrate'], 
                   label=f'{var_num}, d={difficulty}', 
                   marker=markers[j],
                   color=colors[j],
                   linestyle=linestyles[i])

    from matplotlib.lines import Line2D
    
    legend_elements1 = []
    for j, difficulty in enumerate(sorted(difficulties)):
        legend_elements1.append(Line2D([0], [0], color=colors[j], marker=markers[j], label=f'Difficulty={difficulty}', linestyle='-', markersize=8))
    legend_elements2 = []
    
    legend_elements2.append(Line2D([0], [0], color='black', label='50 variables', linestyle='--', markersize=0))
    legend_elements2.append(Line2D([0], [0], color='black', label='100 variables', linestyle='-', markersize=0))
    
    # 合并图例并设置位置
    ax.legend(handles=legend_elements1 + legend_elements2, loc='upper left', frameon=True, framealpha=0.9)

    ax.set_xlabel('Number of solvers')
    ax.set_ylabel('Fork rate of mini-blocks')
    ax.set_xticks(range(1, 21))
    for spine in ax.spines.values():
        spine.set_edgecolor('grey')
    ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
    fig.subplots_adjust(left=0.093, right=0.986, top=0.97, bottom=0.11)
    plt.show()

def plot_bar_chart_kb_forkrate():
    file_path = pathlib.Path.cwd() / "Result_Data\\fig4_20250329\\med_results.json"
    data_list = []
    with open(file_path, 'r') as f:
        jsondata_list = f.read().split('\n')[:-1]
        for jsondata in jsondata_list:
            data_list.append(json.loads(jsondata))
    data_dict = defaultdict()
    for entry in data_list:
        if entry['kb_strategy'] == 'pow':
            kb_strategy = 'hashcash'
        if entry['kb_strategy'] == 'withmini':
            kb_strategy = 'w/ mini-block'
        if entry['kb_strategy'] == 'pow+withmini':
            kb_strategy = 'hashcash + mini-block'
        var_num = entry['var_num']
        difficulty = entry['difficulty']
        miner_num = entry['miner_num']
        # kb_strategy = entry['kb_strategy']
        kb_forkrate = entry['ave_kb_forkrate']
        total_kb_forkrate = entry['total_kb_forkrate']
        # mb_times = entry['mb_times']
        data_dict[(kb_strategy,difficulty)] = kb_forkrate

    # 设置颜色方案为 "Set2"
    colors = plt.cm.Set2.colors

    # 设置字体为 Times New Roman，字号为12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小

    # 从输入字典中提取数据
    difficulties = sorted(list(set(key[1] for key in data_dict.keys())))
    strategies = sorted(list(set(key[0] for key in data_dict.keys())))
    data = np.array(
        [[data_dict[(strategy, difficulty)] for strategy in strategies] 
        for difficulty in difficulties])

    # 绘图
    fig= plt.figure(figsize=(10, 6.5))
    bar_width = 0.2
    x_ticks = np.arange(len(difficulties))

    for i, strategy in enumerate(strategies):
        plt.bar(x_ticks + i * bar_width, 
               data[:, i], 
               bar_width, 
               label=strategy, 
               color=colors[i])
    ax = fig.gca()
    ax.set_xlabel('Difficulty level')
    ax.set_ylabel('Key-block fork rate')
    ax.set_xticks(x_ticks + bar_width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(difficulties)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_fork_rate()
    # plot_bar_chart_kb_forkrate()

