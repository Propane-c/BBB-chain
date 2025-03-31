import time
from typing import Mapping
import matplotlib.pyplot as plt
import json
import os
import xml.etree.ElementTree as ET
import networkx as nx
from pathlib import Path
import tsplib95
import numpy as np
import matplotlib.patches as patches
import pandas as pd
import matplotlib.lines as mlines

_pos:Mapping = None

def read_tsp_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    graph_section = root.find('graph')

    G = nx.Graph()
    pos = {}
    n = len(graph_section)
    distance_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i, vertex in enumerate(graph_section):
        x_tag = vertex.find('x')
        y_tag = vertex.find('y')
        if x_tag is not None and y_tag is not None:
            pos[i] = (float(x_tag.text), float(y_tag.text))
        for edge in vertex.findall('edge'):
            id = int(edge.text)
            cost = float(edge.attrib["cost"])
            distance_matrix[i][id] = cost
            G.add_edge(i, id, weight = cost)
    if len(pos) == 0:
        pos = nx.spring_layout(G, seed=42)
        for i, vertex in enumerate(graph_section):
            pos_x, pos_y = pos[i]
            pos_x_rounded = round(pos_x, 2)
            pos_y_rounded = round(pos_y, 2)
            x_elem = ET.SubElement(vertex, 'x')
            x_elem.text = str(pos_x_rounded)
            y_elem = ET.SubElement(vertex, 'y')
            y_elem.text = str(pos_y_rounded)
        tree.write(file_path)
    global _pos
    _pos = pos
    return G, pos, n, distance_matrix

def draw_tsp_solution(pos, n, opt_x):
    # 遍历解向量 x，添加边到图中
    G = nx.DiGraph()
    # 添加节点
    G.add_nodes_from(range(n))
    opt_x_path = opt_x[:n * n]  # 前 n * n 项是路径变量
    opt_x_reshaped = opt_x_path.reshape((n, n))  # 重塑为 n x n 矩阵
    for i in range(n):
        for j in range(n):
            if opt_x_reshaped[i, j] == 1:  # 如果从城市 i 到城市 j 的路径被选中
                G.add_edge(i, j)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold')
    # 获取prblm_file_path的目录
    output_svg_path = Path.cwd() / "scripts" / "plots" / "tsp_solution.svg"
    plt.savefig(output_svg_path, format="svg")


def plot_solution_progress_tsp(json_dir:str, miner_nums:list, ins:str, ax_main:plt.Axes):
    """绘制TSP求解进度"""
    colors = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#6366f1']
    markers = ['o', 's', '^', 'D', 'P']
    styles = {}
    
    for idx, m in enumerate(miner_nums):
        styles[m] = {
            'color': colors[idx % len(colors)] if ins == "burma14" or (ins == "bayg29" and m <= 3) else colors[idx % len(colors) + 1],
            'marker': markers[idx % len(markers)] if ins == "burma14" or (ins == "bayg29" and m <= 3) else markers[idx % len(markers) + 1],
            'linestyle': '-',
            'alpha': 0.9,
            'zorder': idx + 1
    }
    
    solution_pulp = None
    
    # 创建左上角的子图
    # ax_inset = ax_main.inset_axes([0.52, 0.65, 0.4, 0.28])  # [x, y, width, height]
    # ax_inset2 = ax_main.inset_axes([0.32, 0.65, 0.1, 0.28])  # [x, y, width, height]
    
    # 遍历不同矿工数的结果
    for m in miner_nums:
        json_path = f"{json_dir}/m{m}d5v{ins}evaluation results.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 主图：解的进度
        gas_round_sol_errs = data['gas_round_sol_errs']
        rounds = [item[0] for item in gas_round_sol_errs]
        solutions_bbb = [item[2] for item in gas_round_sol_errs]
        
        if solution_pulp is None:
            solution_pulp = data['solutions_by_pulp'][0]
        
        style = styles[m]
        # 添加渐变填充
        ax_main.fill_between(rounds, solutions_bbb, solution_pulp,
                        color=style['color'], alpha=0.1)
        # 主曲线
        ax_main.plot(rounds, solutions_bbb, 
                color=style['color'],
                marker=style['marker'],
                markersize=4,
                linestyle=style['linestyle'],
                alpha=style['alpha'],
                label=f'{m} solvers',
                zorder=style['zorder'])

    # 主图设置
    ax_main.axhline(y=solution_pulp, color='#414451', linestyle='-.', #D62728
                    linewidth=2, zorder=6)
    if ins == "burma14":
        ax_main.set_xlim([0, 50000])
    elif ins == "bayg29":
        ax_main.set_xlim([0, 50000])
    ax_main.set_xlabel(' ')
    ax_main.set_ylabel(f'Solutions of {ins}',labelpad=20)
    ax_main.grid(True, linestyle='--', alpha=0.3)
    # ax_main.legend(framealpha=0.0, edgecolor='none', fancybox=True,
    #          loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=4)
    if ins == "burma14":
        ax_main.set_ylim(3300, 3850)
    elif ins == "bayg29":
        ax_main.set_ylim(1600, 2000)
    

    colors = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#6366f1']
    miners = [1, 3, 5, 10, 20]
    # 创建色块标注并设置图例
    legend_patches = []
    for i, m in enumerate(miners):
        if ins == "burma14" and m in [1, 3, 5, 10]:
            # 创建小一点的色块
            patch = patches.Patch(color=colors[i % len(colors)], label=f'{m} solvers', alpha=0.8)
            legend_patches.append(patch)
        elif ins == "bayg29" and m in [1, 5, 10, 20]:
            patch = patches.Patch(color=colors[i % len(colors)], label=f'{m} solvers', alpha=0.8)
            legend_patches.append(patch)
        
    # 设置图例，使用handleheight和handlelength参数来控制色块大小
    ax_main.legend(handles=legend_patches, 
                framealpha=0.0, 
                edgecolor='none', 
                fancybox=True,
                loc='upper center', 
                bbox_to_anchor=(0.5, 1.01), 
                ncol=5, 
                frameon=False,
                handlelength=1.0,  # 减小色块宽度
                handleheight=0.5)  # 减小色块高度
    # 在左上角子图中绘制gas和error
    # plot_gas_vs_round(json_dir, miner_nums, ax_inset)
    # plot_ave_solution_error_vs_round(json_dir, miner_nums, ax_inset2)
    
    # 移除子图多余的刻度标签
    # ax_inset.set_xticklabels([])

def plot_ave_solution_error_vs_round(json_dir, miner_nums:list, ins:str, ax_error:plt.Axes):
    """绘制平均解误差"""
    colors = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#6366f1']
    markers = ['o', 's', '^', 'D', 'P']
    styles = {}
    
    for idx, m in enumerate(miner_nums):
        styles[m] = {
            'color': colors[idx % len(colors)],
            'marker': markers[idx % len(markers)],
            'linestyle': '-',
            'alpha': 0.9,
            'zorder': idx + 1
        }
    
    for m in miner_nums:
        json_path = f"{json_dir}/m{m}d5v{ins}evaluation results.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        gas_round_sol_errs = data['gas_round_sol_errs']
        rounds = [item[0] for item in gas_round_sol_errs]
        solution_errs = [item[3] for item in gas_round_sol_errs]
        
        style = styles[m]
        ax_error.plot(rounds, solution_errs,
                color=style['color'],
                marker=style['marker'],
                markersize=3,
                linewidth=1.5,
                alpha=style['alpha'],
                zorder=style['zorder'])
    # 误差子图设置
    ax_error.set_xlabel(' ')
    ax_error.set_ylabel('Error')
    ax_error.set_xlim([0, 10000])
    ax_error.grid(True, linestyle='--', alpha=0.3)
    ax_error.set_yscale('log')
    for spine in ax_error.spines.values():
        spine.set_edgecolor('#dddddd')
    

def visualize_tsp_with_tsplib(round_num, json_path, miner_num=None, ins:str="burma14", ax=None):
    """使用tsplib数据可视化TSP问题"""
    # 加载TSP问题实例
    problem = tsplib95.load(f"E:\Files\gitspace\\bbb-github\\tsp_origin\\sourcesSymmetricTSP\\{ins}.tsp")
    if ax is None:
        # 创建图形
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(1, 1, 1)
    # 创建NetworkX图
    G = problem.get_graph()
    print(problem.as_name_dict())
    
    # 获取节点坐标
    coords = problem.display_data if ins == "bayg29" else problem.node_coords
    print(coords)
    pos = {i-1: (coords[i][0], coords[i][1]) for i in G.nodes()}
    
    # 如果指定了round_num和json_path，读取对应轮次的解
    with open(json_path, 'r') as f:
        data = json.load(f)
        gas_round_sol_errs = data['gas_round_sol_errs']
        for item in gas_round_sol_errs:
            if item[0] == round_num:  # 找到对应轮次
                opt_x = item[4]  # ix在第5个位置
                break
    
    # 创建图形
    G = nx.DiGraph()
    # 添加节点
    n = 14 if ins == "burma14" else 29
    # opt_x = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 1.0, -0.0, 0.0, 0.0, 0.0, 1.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 1.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 13.0, 11.0, 7.0, 9.0, 2.0])
    G.add_nodes_from(range(n))
    
    opt_x_path = opt_x[:n * n]  # 前 n * n 项是路径变量
    opt_x_reshaped = np.array(opt_x_path).reshape((n, n))  # 重塑为 n x n 矩阵
    for i in range(n):
        for j in range(n):
            if opt_x_reshaped[i, j] == 1:  # 如果从城市 i 到城市 j 的路径被选中
                G.add_edge(i, j)
    ax.set_facecolor('#f8f9fa') 
    # 绘制图
    nx.draw(G, pos, with_labels=False, 
            node_size=150,
            node_color='#6366f1',  # 节点不填充颜色
            # edgecolors='#6366f1',  # 节点边框颜色
            linewidths=1,  # 节点边框宽度
            ax=ax,
            alpha=0.8,
            edge_color='#4A4A4A', 
            width=2, 
            arrowsize=20,
            arrowstyle='-|>',  # 箭头样式
            arrows=True,  # 显示箭头
            style='dashed')
    

    # 显示坐标轴
    ax.set_axis_on()
    
    # 设置坐标轴范围和刻度
    x_coords = [pos[i][0] for i in pos]
    y_coords = [pos[i][1] for i in pos]
    margin = 200 if ins == "bayg29" else 0.8
    
    # 设置x轴范围和刻度
    x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
    ax.set_xlim(x_min, x_max)
    
    # 设置y轴范围和刻度
    y_min, y_max = min(y_coords) - 0.5*margin, max(y_coords) + 0.5*margin
    ax.set_ylim(y_min, y_max)
    for spine in ax.spines.values():
        spine.set_edgecolor('#dddddd')
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)
    # plt.tight_layout()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    
    plt.savefig(f"E:\Files\A-blockchain\\branchbound\\figs\\tsp\\{ins}m{miner_num}r{round_num}_{time.strftime('%Y%m%d%H%M%S')}.svg", dpi=300)
    # plt.show()
    return G, pos

    

def plot_gas_vs_round2(json_dir, miner_nums:list, ins:str, ax_gas:plt.Axes):
    """绘制gas随round的变化"""
    colors = ['#3b82f6', '#10b981', '#ef4444', '#f59e0b', '#6366f1']
    markers = ['o', 's', '^', 'D', 'P']
    styles = {}
    
    for idx, m in enumerate(miner_nums):
        styles[m] = {
            'color': colors[idx % len(colors)] if ins == "burma14" or (ins == "bayg29" and m <= 3) else colors[idx % len(colors) + 1],
            'marker': markers[idx % len(markers)] if ins == "burma14" or (ins == "bayg29" and m <= 3) else markers[idx % len(markers) + 1],
            'linestyle': '-',
            'alpha': 0.5,
            'zorder': len(miner_nums) - idx
        }
    ax_gas2 = ax_gas.twinx()
    for m in miner_nums:
        json_path = f"{json_dir}/m{m}d5v{ins}evaluation results.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        gas_consumes = data['gas_consumes']
        rounds = [item[0] for item in gas_consumes]
        print("rounds",len(rounds))
        solve_rounds = data['solve_rounds']
        max_round = solve_rounds["B0"] if solve_rounds else 0

        gas_diffs = [0] * (max_round + 1) 
        prev_gas = None
        for item in gas_consumes:
            round_num = item[0]
            curr_gas = item[1]
            if prev_gas is not None and round_num <= max_round:
                gas_diff = curr_gas - prev_gas
                gas_diffs[round_num] = gas_diff
            prev_gas = curr_gas
        window = 50 if ins == "burma14" else 100
        gas_diffs_smooth = pd.Series(gas_diffs).rolling(window=window, center=True).mean()
        gas_diffs_smooth = gas_diffs_smooth.fillna(method='bfill').fillna(method='ffill')
        rounds_diff = list(range(len(gas_diffs_smooth)))

        style = styles[m]
        ax_gas.fill_between([0, max_round], [8, 8], 0,
                          color=style['color'], alpha=0.05, zorder=1)
        ax_gas.fill_between(rounds_diff, gas_diffs_smooth, 0,
                         color=style['color'], alpha=0.6, rasterized=True, zorder=style['zorder'])
        ax_gas.axvline(x=max_round, color='#4b5563', linestyle='--', linewidth=0.5, alpha=0.8, zorder=2)
        # # 绘制曲线
        # ax_gas.plot(rounds_diff, gas_diffs_smooth, 
        #         color=style['color'],
        #         linewidth=1,  # 增加线宽
        #         linestyle=style['linestyle'],
        #         alpha=style['alpha'],
        #         label=f'{m} miners (diff)',
        #         zorder=style['zorder'])
        gas_consumes = data['gas_consumes']
        rounds = [item[0] for item in gas_consumes]
        gas_values = [item[1] for item in gas_consumes]
        style = styles[m]
        ax_gas2.plot(rounds, gas_values,
                color=style['color'],
                linewidth=1,
                linestyle='-',
                marker=style['marker'],
                markersize=3,
                markevery=0.05,
                alpha=1,
                zorder=2)
    # 设置左轴
    ax_gas.set_ylabel(f'Gas consumption\n per round of {ins}',labelpad=25)
    ax_gas.set_ylim(0, 8)
    ax_gas.set_xlabel("Round")
    if ins == "burma14":
        ax_gas.set_xlim([0, 50000])
        ax_gas2.set_ylim(0, 25000)
        ax_gas2.set_yticks([0, 20000])
    elif ins == "bayg29":
        ax_gas.set_xlim([0, 50000])
        ax_gas2.set_ylim(0, 60000)
        ax_gas2.set_yticks([0, 60000])
    for spine in ax_gas2.spines.values():
        spine.set_edgecolor('grey')
    # ax_gas2.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
    ax_gas2.tick_params(axis='y', rotation=90)
    
    ax_gas2.set_ylabel("Total gas consumption")
    ax_gas.grid(True, linestyle='--', alpha=0.2)
    
    
    return ax_gas

def plot_case_tsp():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14


    fig = plt.figure(figsize=(10, 12))  # 调整整体图大小
    gs = fig.add_gridspec(4, 1,  height_ratios=[5, 2, 5, 2], width_ratios=[1])  # 添加间距控制
    
    ax_a = fig.add_subplot(gs[0, 0])
    json_dir = "E:\Files\gitspace\\bbb-github\Results\\20250312\\202035"
    plot_solution_progress_tsp(json_dir, miner_nums=[1, 3, 5, 10], ins = "burma14", ax_main=ax_a)
    ax_b = fig.add_subplot(gs[1, 0])
    plot_gas_vs_round2(json_dir, miner_nums=[1, 3, 5, 10], ins = "burma14", ax_gas=ax_b)
    
    ax_c = fig.add_subplot(gs[2, 0])
    json_dir2 = "E:\Files\gitspace\\bbb-github\Results\\20250320"
    plot_solution_progress_tsp(json_dir2, miner_nums=[1, 5, 10, 20], ins = "bayg29", ax_main=ax_c)
    
    ax_d = fig.add_subplot(gs[3,0])
    plot_gas_vs_round2(json_dir2, miner_nums=[1, 5, 10, 20], ins = "bayg29", ax_gas=ax_d)
    
    # ax_e = fig.add_subplot(gs[4,0])
    # plot_gas_vs_round1(json_dir, miner_nums=[1, 3, 5, 10], ins = "burma14", ax_gas=ax_e)
    # plot_gas_vs_round1(json_dir, json_dir2, miner_nums=[1, 3, 5, 10], miner_nums2=[1, 5, 10, 20], ax_gas=ax_e)
    ax_list = [ax_a, ax_b, ax_c, ax_d]
    for ax in ax_list:
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
    
    # 调整标签位置
    labels = ['a', 'b', 'c', 'd']
    axes = [ax_a, ax_b, ax_c, ax_d]
    for ax, label in zip(axes, labels):
        ax.text(-0.15, 1.01, label, transform=ax.transAxes, 
                fontsize=16, fontweight='bold')
    
    fig.subplots_adjust(left=0.13, bottom=0.05, right=0.93, top=0.979, hspace=0.22)
    # plt.tight_layout()
    plt.savefig(f"E:\Files\A-blockchain\\branchbound\\figs\\tsp{time.strftime('%Y%m%d%H%M%S')}.svg", dpi=300)
    # plt.show()

if __name__ == "__main__":
    # 使用原有的XML方法
    # load_file_path = Path.cwd() / "tsp_original" / "tsp_original.xml"
    # G, pos, n, distance_matrix = read_tsp_from_xml(load_file_path)
    # draw_tsp_solution(_pos, n, opt_x)
    
    # 使用新的tsplib95方法
    # tsp_file = Path.cwd() / "tsp_origin" / "tsp" / "burma14.tsp"
    # G_tsp, pos_tsp = visualize_tsp_with_tsplib(tsp_file)
    plot_case_tsp()
    
    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250312\\202035\m1d5vburma14evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=525, json_path=json_path, miner_num=1)
    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250312\\202035\m5d5vburma14evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=1808, json_path=json_path, miner_num=5)
    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250312\\202035\m1d5vburma14evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=2335, json_path=json_path, miner_num=1)
    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250312\\202035\m10d5vburma14evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=1826, json_path=json_path, miner_num=10)
    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250312\\202035\m3d5vburma14evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=16485, json_path=json_path, miner_num=3)


    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250320\m1d5vbayg29evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=3491, json_path=json_path, ins = "bayg29", miner_num=1)
    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250320\m10d5vbayg29evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=1704, json_path=json_path, ins = "bayg29", miner_num=10)
    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250320\m20d5vbayg29evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=7317, json_path=json_path, ins = "bayg29", miner_num=20)
    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250320\m10d5vbayg29evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=4825, json_path=json_path, ins = "bayg29", miner_num=10)
    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250320\m5d5vbayg29evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=16546, json_path=json_path, ins = "bayg29", miner_num=5)
    
