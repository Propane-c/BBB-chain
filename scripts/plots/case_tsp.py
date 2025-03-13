from typing import Mapping
import matplotlib.pyplot as plt
import json
import os
import xml.etree.ElementTree as ET
import networkx as nx
from pathlib import Path
import tsplib95
import numpy as np

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


def plot_solution_progress_tsp(json_dir:str, miner_nums:list, ax_main:plt.Axes):
    """绘制TSP求解进度"""
    # 使用更丰富的配色方案
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
    
    solution_pulp = None
    
    # 遍历不同矿工数的结果
    for m in miner_nums:
        json_path = f"{json_dir}/m{m}d5vburma14evaluation results.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
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
                label=f'{m} miners',
                linewidth=1.5,  # 增加线宽
                zorder=style['zorder'])
    
    # 主图设置
    ax_main.axhline(y=solution_pulp, color='#4b5563', linestyle='--', 
                    linewidth=1.5, zorder=1)
    ax_main.set_xlim([0, 30000])
    ax_main.set_xlabel('Round')
    ax_main.set_ylabel('Solutions of Burma14')
    ax_main.grid(True, linestyle='--', alpha=0.8, zorder=0)
    
    # 优化图例
    ax_main.legend(framealpha=0.8, edgecolor='grey', fancybox=True, 
                loc='center right', ncol=1)
    
    # 设置y轴范围并添加边距
    ax_main.set_ylim(3275, 3650)

def plot_ave_solution_error_vs_round(json_dir, miner_nums:list, ax_error:plt.Axes):
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
        json_path = f"{json_dir}/m{m}d5vburma14evaluation results.json"
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
                markevery=0.1,
                linewidth=1.5,
                alpha=style['alpha'],
                zorder=style['zorder'])
    # 误差子图设置
    ax_error.set_xlabel('Round')
    ax_error.set_ylabel('Error')
    ax_error.set_xlim([0, 30000])
    ax_error.grid(True, linestyle='--', alpha=0.3)
    ax_error.set_yscale('log')
    

def visualize_tsp_with_tsplib(round_num=None, json_path=None, ax:plt.Axes=None):
    """使用tsplib95加载和可视化TSP实例
    
    Args:
        file_path: TSP实例文件路径
        round_num: 要显示的轮次
        json_path: 包含解的json文件路径
    """
    # 加载TSP问题实例
    problem = tsplib95.load("E:\Files\gitspace\\bbb-github\\tsp_origin\\sourcesSymmetricTSP\\burma14.tsp")
    if ax is None:
        # 创建图形
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(1, 1, 1)
    # 创建NetworkX图
    G = problem.get_graph()
    print(problem.as_name_dict())
    
    # 获取节点坐标
    coords = problem.node_coords
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
    n = 14
    # opt_x = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 1.0, -0.0, 0.0, 0.0, 0.0, 1.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 1.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 13.0, 11.0, 7.0, 9.0, 2.0])
    G.add_nodes_from(range(n))
    
    opt_x_path = opt_x[:n * n]  # 前 n * n 项是路径变量
    opt_x_reshaped = np.array(opt_x_path).reshape((n, n))  # 重塑为 n x n 矩阵
    for i in range(n):
        for j in range(n):
            if opt_x_reshaped[i, j] == 1:  # 如果从城市 i 到城市 j 的路径被选中
                G.add_edge(i, j)
    
    # 绘制图
    nx.draw(G, pos, with_labels=False, node_size=150, 
            node_color='#6366f1',  ax=ax,  # 使用暖色调的橙色
            alpha=0.8,
            edge_color='#4A4A4A', width=2, arrowsize=20, 
            arrowstyle='-|>', style='dashed')
    
    # 显示坐标轴
    ax.set_axis_on()
    
    # 设置坐标轴范围和刻度
    x_coords = [pos[i][0] for i in pos]
    y_coords = [pos[i][1] for i in pos]
    margin = 0.8
    
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
    plt.tight_layout()
    plt.show()
    return G, pos

def plot_gas_vs_round(json_dir, miner_nums:list, ax_gas:plt.Axes):
    """绘制gas随round的变化"""
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
        json_path = f"{json_dir}/m{m}d5vburma14evaluation results.json"
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        gas_consumes = data['gas_consumes']
        rounds = [item[0] for item in gas_consumes]
        cur_gas = [20000-item[1] for item in gas_consumes]
        
        style = styles[m]
        ax_gas.plot(rounds, cur_gas, 
                color=style['color'],
                linewidth=1.5,
                linestyle=style['linestyle'],
                alpha=style['alpha'],
                label=f'{m} miners',
                zorder=style['zorder'])
    
    # 设置坐标轴
    ax_gas.set_xlabel('Round')
    ax_gas.set_ylabel('Rest gas')
    ax_gas.set_xlim([0, 30000])
    ax_gas.grid(True, linestyle='--', alpha=0.3)
    # ax_gas.legend(framealpha=0.0, edgecolor='none', fancybox=True)

def plot_case_tsp():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14


    fig = plt.figure(figsize=(10, 7))  # 调整整体图大小
    gs = fig.add_gridspec(2, 2, height_ratios=[3.5, 1], width_ratios=[2, 1],hspace=0.2, wspace=0.25)  # 添加间距控制
    
    ax_b = fig.add_subplot(gs[0, 0:2])
    json_dir = "E:\Files\gitspace\\bbb-github\Results\\20250312\\202035"
    plot_solution_progress_tsp(json_dir, miner_nums=[1, 3, 5, 10], ax_main=ax_b)

    ax_c = fig.add_subplot(gs[1, 0])
    plot_ave_solution_error_vs_round(json_dir, miner_nums=[1, 3, 5, 10], ax_error=ax_c)
    
    # ax_d = fig.add_subplot(gs[1, 1])
    # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250312\\191401\m1d5vburma14evaluation results.json"
    # visualize_tsp_with_tsplib(round_num=14412, json_path=json_path, ax=ax_d)
    
    ax_e = fig.add_subplot(gs[1,1])
    plot_gas_vs_round(json_dir, miner_nums=[1, 3, 5, 10], ax_gas=ax_e)
    
    ax_list = [ax_b, ax_c, ax_e]
    for ax in ax_list:
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
    
    # 调整标签位置
    labels = ['a', 'b', 'c', 'd', 'e', 'f']
    axes = [ax_b, ax_c, ax_e]
    for ax, label in zip(axes, labels):
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, 
                fontsize=16, fontweight='bold')
    
    fig.subplots_adjust(left=0.095, bottom=0.074, right=0.98, top=0.96, hspace=0.4, wspace=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 使用原有的XML方法
    # load_file_path = Path.cwd() / "tsp_original" / "tsp_original.xml"
    # G, pos, n, distance_matrix = read_tsp_from_xml(load_file_path)
    # draw_tsp_solution(_pos, n, opt_x)
    
    # 使用新的tsplib95方法
    # tsp_file = Path.cwd() / "tsp_origin" / "tsp" / "burma14.tsp"
    # G_tsp, pos_tsp = visualize_tsp_with_tsplib(tsp_file)
    # plot_case_tsp()
    json_path = "E:\Files\gitspace\\bbb-github\Results\\20250312\\202035\m10d5vburma14evaluation results.json"
    visualize_tsp_with_tsplib(round_num=4873, json_path=json_path)
    json_path = "E:\Files\gitspace\\bbb-github\Results\\20250312\\202035\m1d5vburma14evaluation results.json"
    visualize_tsp_with_tsplib(round_num=13565, json_path=json_path)
    json_path = "E:\Files\gitspace\\bbb-github\Results\\20250312\\202035\m5d5vburma14evaluation results.json"
    visualize_tsp_with_tsplib(round_num=919, json_path=json_path)
