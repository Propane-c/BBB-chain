import json
import math
import pathlib
import time
from collections import defaultdict
import sys
sys.path.append('E:/Files/gitspace/bbb-github')

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.patches import (
    Ellipse,
    Rectangle,
)
from matplotlib.path import Path
from matplotlib.ticker import PercentFormatter
from scipy.signal import find_peaks, peak_prominences

import branchbound.bb_consensus as bb

SAVE_PREFIX = "E:\Files\A-blockchain\\branchbound\\branchbound仿真\\0415"
pathlib.Path.mkdir(pathlib.Path(SAVE_PREFIX), exist_ok=True)
SAVE = True

MAXSAT='maxsat'
TSP='tsp'
MIPLTP='miplib'

def plot_feasible_regions_fig1(FIG_NUM):
    Path.mkdir(Path(SAVE_PREFIX), exist_ok=True)
    print("FIUUG_NUM: ")
    FIG_NUM = int(input())
    # FIG_NUM = 5
    public = {'c' : np.array([15, -8]),
            'G_ub' : np.array([[-1,0], [0,-1], [-18, 8], [-20, 17], [3, 1]]),
            'h_ub' : np.array([0,0, -15, -2, 17]),}

    extra1 = {'G_ub': [[0,-1]],
            'h_ub': [-2],
            'x_lp': [ 1.800e+00,  2.000e+00],
            'z_lp': -11,}

    extra2 = {'G_ub': [[0,1]],
            'h_ub': [1],
            'x_lp': [1.278e+00,  1.000e+00],
            'z_lp': -11.167,}
    if FIG_NUM == 2:
        public['G_ub'] = np.append(public['G_ub'], [[0,-1]], axis = 0)
        public['h_ub'] = np.append(public['h_ub'], [-2], axis = 0)
        extra1 = {'G_ub': [[-1,0]],
            'h_ub': [-2],
            'x_lp': [ 2.000e+00,  2.235e+00],
            'z_lp': -12.118,}
        extra2 = {'G_ub': [[1,0]],
            'h_ub': [1],
            'x_lp': None,
            'z_lp': -10.089,}
    
    elif FIG_NUM == 3:
        public['G_ub'] = np.append(public['G_ub'], [[0,1]], axis = 0)
        public['h_ub'] = np.append(public['h_ub'], [1], axis = 0)
        extra1 = {'G_ub': [[-1,0]],
            'h_ub': [-2],
            'x_lp': [2.000e+00,  1.000e+00],
            'z_lp': -22,}
        extra2 = {'G_ub': [[1,0]],
            'h_ub': [1],
            'x_lp': [1.000e+00,  3.750e-0],
            'z_lp': -12,}
        
    elif FIG_NUM == 4:
        public['G_ub'] = np.append(public['G_ub'], [[0,-1],[-1,0]], axis = 0)
        public['h_ub'] = np.append(public['h_ub'], [-2,-2], axis = 0)
        extra1 = {'G_ub': [[0,-1]],
            'h_ub': [-3],
            'x_lp': [2.650e+00,  3.000e+00],
            'z_lp': -15.75,}
        extra2 = {'G_ub': [[0,1]],
            'h_ub': [2],
            'x_lp': [2.000e+00,  2.000e+00],
            'z_lp': -14,}
    
    elif FIG_NUM == 5:
        public['G_ub'] = np.append(public['G_ub'], [[0,1],[1,0]], axis = 0)
        public['h_ub'] = np.append(public['h_ub'], [1,1], axis = 0)
        extra1 = {'G_ub': [[0,-1]],
            'h_ub': [-1],
            'x_lp': None,
            'z_lp': -22,}
        extra2 = {'G_ub': [[0,1]],
            'h_ub': [0],
            'x_lp': [8.333e-01, -0.000e+00],
            'z_lp': -12.5,}
    
    elif FIG_NUM == 6:
        public['G_ub'] = np.append(public['G_ub'], [[0,1],[1,0],[0,1]], axis = 0)
        public['h_ub'] = np.append(public['h_ub'], [1,1,0], axis = 0)
        extra1 = {'G_ub': [[-1,0]],
            'h_ub': [-1],
            'x_lp': [1.000e+00, -0.000e+00],
            'z_lp': -15,}
        extra2 = {'G_ub': [[1,0]],
            'h_ub': [0],
            'x_lp': None,
            'z_lp': -12.5,}
    

    # 设置绘图范围
    x_range = np.linspace(0, 6, 400)
    y_range = np.linspace(0, 5, 400)
    if FIG_NUM in [2]:
        x_range = np.linspace(0.3, 5.2, 400)
        y_range = np.linspace(1.6, 5, 400)
    if FIG_NUM in [3]:
        x_range = np.linspace(0, 6, 400)
        y_range = np.linspace(-0.3, 1.5, 400)
    if FIG_NUM in [4]:
        x_range = np.linspace(1.5, 5.2, 400)
        y_range = np.linspace(1.8, 5, 400)
    if FIG_NUM in [5]:
        x_range = np.linspace(0.5, 1.2, 400)
        y_range = np.linspace(-0.1, 1.2, 400)
    if FIG_NUM in [6]:
        x_range = np.linspace(-0.3, 1.2, 400)
        y_range = np.linspace(-0.1, 0.7, 400)
    # B3
    # x_range = np.linspace(-1, 5, 400)
    # y_range = np.linspace(-1, 4, 400)
    # B5
    # x_range = np.linspace(-5, 3, 400)
    # y_range = np.linspace(-9, 2, 400)
    X, Y = np.meshgrid(x_range, y_range)
    if FIG_NUM not in [3, 5]:
        fig = plt.figure(figsize=(7, 7))
    elif FIG_NUM == 3:
        fig = plt.figure(figsize=(7, 3))
    elif FIG_NUM == 5:
        fig = plt.figure(figsize=(4.5, 7))
    
    def draw_lp2D(public_pblm:dict, ex1:dict, ex2:dict):
        # 绘制线性规划问题的解和约束条件，但不绘制等高线
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        if FIG_NUM in [3,5,6]:
            plt.rcParams['font.size'] = 30  # 调整字号大小
        else:
            plt.rcParams['font.size'] = 40  # 调整字号大小

        colors = [(31,119,180), (255,127,14), (44,160,44), (214,39,40),(214,39,40),(214,39,40),(148,103,189),
                   (148,103,189), (214,39,40), (148,103,189),
                  (214,39,40), (148,103,189),(214,39,40), (148,103,189),(214,39,40), (148,103,189),
                 (140,86,75), (227,119, 194), (127,127,127), (188,189,34), (23,1990,207),(214,39,40),(148,103,189),]
        colors = [(r/255.0, g/255.0, b/255.0) for r, g, b in colors]
        colors.insert(0,'black')
        colors.insert(0,'black')
        cid = 0

        def feasible_region(G_ub, h_ub, label, pos_x, pos_y, region_color, font_color):
            # 计算每个约束条件下y的值，并创建一个遮罩表示可行域
            print(G_ub, h_ub)
            feasible_mask = np.ones(X.shape, dtype=bool)
            for (a, b), h in zip(G_ub, h_ub):
                if a!= 0 and b != 0:
                    feasible_mask &= (a*X + b*Y <= h)
                elif b == 0:
                    feasible_mask &= (a*X <= h)
                elif a == 0:
                    feasible_mask &= (b*Y <= h)
            alpha = 0.2 if FIG_NUM == 0 else 0.5
            plt.imshow(feasible_mask, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), 
                    origin='lower', cmap=region_color, alpha=alpha, aspect='auto',zorder=0)
            plt.text(pos_x, pos_y, label, fontsize=60, color=font_color, fontweight='bold',style='italic')

        c = public_pblm['c']
        G_ub1 = np.append(public_pblm['G_ub'], ex1['G_ub'], axis=0)
        h_ub1 = np.append(public_pblm['h_ub'], ex1['h_ub'])
        G_ub2 = np.append(public_pblm['G_ub'], ex2['G_ub'], axis=0)
        h_ub2 = np.append(public_pblm['h_ub'], ex2['h_ub'])

        # 绘制不等式约束的边界和可行域
        if FIG_NUM == 0:
            feasible_region(public_pblm['G_ub'],public_pblm['h_ub'], " ", 2.1, 1.2,"Greens", "green")
        elif FIG_NUM == 1:
            feasible_region(G_ub1,h_ub1, "1+", 2.9, 2.7,"Oranges", "#ff8c00")
            feasible_region(G_ub2,h_ub2, "1-", 2.6, 0.3,"Blues", (31/255, 119/255, 180/255))
        elif FIG_NUM == 2:
            feasible_region(G_ub1,h_ub1, "2+", 3.2, 2.8,"Oranges", "#ff8c00")
            feasible_region(ex2['G_ub'],ex2['h_ub'], "", 0.5, 3.0,"Greys", (31/255, 119/255, 180/255))
        elif FIG_NUM == 3:
            feasible_region(G_ub1,h_ub1, "  3+", 2.6, 0.2,"Oranges", "#ff8c00")
            feasible_region(G_ub2,h_ub2, "3-", 0.2, 0.7,"Blues", (31/255, 119/255, 180/255))
            
            # plt.text(2.6, 0.2, "Integer", fontsize=40, color='r', fontweight='bold',style='italic')
            plt.annotate(" ",
                    xy=(0.9,0.1),
                    xytext=(-20, 40),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color='r', linewidth=4),
                    color='r',
                    fontsize = 16,)
        elif FIG_NUM == 4:
            feasible_region(G_ub1,h_ub1, "4+", 3.4, 3.3,"Oranges", "#ff8c00")
            feasible_region(G_ub2,h_ub2, "4-", 2.8, 2.2,"Blues", (31/255, 119/255, 180/255))
            # plt.text(3.2, 3.3, "Integer", fontsize=60, color='r', fontweight='bold',style='italic')
            ax = fig.gca()
            ax.add_patch(Ellipse((3.5, 2), width=3, height=0.2, 
                        edgecolor=(31/255,119/255,180/255), facecolor='none', linestyle='--', linewidth=4))
        elif FIG_NUM == 5:
            feasible_region(ex1['G_ub'],ex1['h_ub'], "", 0.7, 1.05,"Greys", "#ff8c00")
            feasible_region(G_ub2,h_ub2, "5-", 0.8, 0.25,"Blues", (31/255, 119/255, 180/255))
            ax = fig.gca()
            ax.add_patch(Ellipse((0.91, 0), width=0.2, height=0.1, 
                        edgecolor=(31/255,119/255,180/255), facecolor='none', linestyle='--', linewidth=4))
            # plt.annotate(" ",
            #         xy=(0.9,0.05),
            #         xytext=(-20, 50),
            #         textcoords="offset points",
            #         arrowprops=dict(arrowstyle="->", color='b', linewidth=2),
            #         color=(31/255,119/255,180/255),
            #         fontsize = 16,)
        elif FIG_NUM == 6:
            feasible_region(G_ub1,h_ub1, "6+", 0.6, -5.2,"Oranges", "#ff8c00")
            feasible_region(ex2['G_ub'],ex2['h_ub'], "", -0.2, 0.25,"Greys", (31/255, 119/255, 180/255))
            plt.annotate("6+",
                    xy=(1,0),
                    xytext=(-150, 100),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="#ff8c00", linewidth=4),
                    color="#ff8c00",
                    fontsize = 60,
                    fontweight = 'bold',
                    fontstyle='italic')

        def draw_boundary(G_ub, h_ub, isPublic:bool=False):
            nonlocal cid
            
            lw = 1 if isPublic else 1
            for i, ((x, y), h) in enumerate(zip(G_ub, h_ub)):
                print("cid: ", cid-2)
                # 计算约束边界线的y值
                if x!=0 and y != 0:
                    y_boundary = (h - x*x_range) / y
                    label = f'{x}x+{y}y<={h}' if not isPublic else None
                    plt.plot(x_range, y_boundary, lw=lw, label = label,color = colors[cid])
                elif x==0 and y != 0:
                    y_boundary = (h - x*x_range) / y
                    label = f"{-y}y ≥ {-h}" if y<0 else f'{y}y ≤ {h}'
                    label = f"y ≥ {-h}" if y == -1 else label
                    label = f"y ≤ {h}" if y == 1 else label
                    label = label if not isPublic else None
                    plt.plot(x_range, y_boundary, lw=lw, label = label,color = colors[cid])
                else:
                    label = f"{-x}x ≥ {-h}" if x<0 else f'{x}x ≤ {h}'
                    label = f"x ≥ {-h}" if x == -1 else label
                    label = f"x ≤ {h}" if x == 1 else label
                    label = label if not isPublic else None
                    plt.axvline(x=h/x if x != 0 else 0, lw=lw, label = label,color = colors[cid])  # 垂直线
                cid+=1
        if FIG_NUM == 0:
            draw_boundary(public_pblm['G_ub'], public_pblm['h_ub'],False)
        else:
            draw_boundary(public_pblm['G_ub'], public_pblm['h_ub'],True)
            draw_boundary(ex1['G_ub'], ex1['h_ub'])
            draw_boundary(ex2['G_ub'], ex2['h_ub'])

        def mark_opt(x_lp:list, z_lp:float, pos_x, pos_y, id:int, isOpt=False):
            # 标记最优解的点
            if x_lp is None:
                return
            if isOpt:
                mark = 'r*'
                size = 50
            else:
                mark = 'ro'
                size = 15
            plt.plot(x_lp[0], x_lp[1], mark ,markersize = size)  # 红色圆圈表示最优点
            # plt.annotate(f"Optimal point {id}\n({x_lp[0]}, {x_lp[1]})\nvalue = {z_lp}",
            # plt.annotate(f"Optimal{id}",
            #     xy=(x_lp[0]-0.02, x_lp[1]+0.02),
            #     xytext=(pos_x, pos_y),
            #     textcoords="offset points",
            #     arrowprops=dict(arrowstyle="->", color='r', linewidth=2),
            #     color='r',
            #     fontsize = 30,
            #     fontweight='bold')
            # plt.text(pos_x, pos_y, 
            #         f"Point ({x_lp[0]}, {x_lp[1]})\nwith value = {z_lp}", 
            #         verticalalignment='bottom', horizontalalignment='left')
            # opt = c @ x_lp
            # slope = -c[0] / c[1]
            # y_opt = slope * (x_range - x_lp[0]) + x_lp[1]
            # plt.plot(x_range, y_opt, 'r--', label=f'Optimal 15x-8y={opt}')
        if FIG_NUM == 0:
            mark_opt(ex1['x_lp'], ex1['z_lp'], -80,55, '')
        elif FIG_NUM == 1:
            mark_opt(ex1['x_lp'], ex1['z_lp'], -100,35, '+')
            mark_opt(ex2['x_lp'], ex2['z_lp'], 80,15, "-")
        elif FIG_NUM == 2:
            mark_opt(ex1['x_lp'], ex1['z_lp'], -120,25, '+')
            mark_opt(ex2['x_lp'], ex2['z_lp'],-170,-55, "-")
        elif FIG_NUM == 3:
            mark_opt(ex1['x_lp'], ex1['z_lp'], 20,55, '+')
            mark_opt(ex2['x_lp'], ex2['z_lp'],170,-100, "-")
        elif FIG_NUM == 4:
            mark_opt(ex1['x_lp'], ex1['z_lp'], 20,-55, '+', True)
            mark_opt(ex2['x_lp'], ex2['z_lp'],-120,20, "-")
        elif FIG_NUM == 5:
            mark_opt(ex1['x_lp'], ex1['z_lp'], 20,-55, '+')
            mark_opt(ex2['x_lp'], ex2['z_lp'],20,-70, "-")
        elif FIG_NUM == 6:
            mark_opt(ex1['x_lp'], ex1['z_lp'], 20,-55, '+')
            mark_opt(ex2['x_lp'], ex2['z_lp'],20,-70, "-")

        # 图表设置
        # plt.title('Linear Programming')
        # plt.xlabel('x')
        # plt.ylabel('y')
        plt.grid(True)
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlim(x_range.min(), x_range.max())
        plt.ylim(y_range.min(), y_range.max())
        # plt.legend(loc = "upper left",prop={'family': 'Times New Roman'})
        plt.savefig(SAVE_PREFIX+F"\\B{FIG_NUM}_2.svg")
        plt.show()
        plt.close()
    draw_lp2D(public, extra1, extra2)


def plot_bounds_fig1(data_list:list[dict]):
    # 读取并解析文件
    # with open(file_path_new, 'r') as file:
    #     json_data_new = json.load(file)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小

    def get_pname(pname_list:list):
        if pname_list == "None":
            return None
        return tuple(tuple(x) if isinstance(x, list) else x for x in pname_list)

    def count_children(ub_data):
        # 更新每个点的后续点数量
        for point in ub_data:
            pre_pname = get_pname(point["pre_pname"])
            # 遍历所有点，增加前序点的后续点计数
            while pre_pname:
                children_counts[pre_pname] += 1
                # 找到前序点的前序点
                pre_point = point_lookup.get(pre_pname)
                if pre_point is None:
                    break
                pre_pname = get_pname(pre_point["pre_pname"])
        return children_counts

    data = data_list[0]
    ub_data = data.get("ubdata", [])
    lowerbounds = data.get("lowerbounds", {})
    point_lookup = {get_pname(point['pname']): point for point in ub_data}
    children_counts = defaultdict(lambda : 1) 
    count_children(ub_data)

    rounds = []
    brounds = []
    ubs = []
    pnames = []
    pre_pnames = []
    children_counts = defaultdict(lambda : 1) 
    points_by_mb = defaultdict(list)

    # 使用更新后的函数计算后续点数量
    count_children(data["ubdata"])

    for point in ub_data:
        pname = get_pname(point["pname"])
        pre_pname = get_pname(point["pre_pname"])
        rounds.append(point["round"])
        brounds.append(point["bround"])
        ubs.append(point["ub"])
        pnames.append(pname)
        pre_pnames.append(pre_pname)
        points_by_mb[point["block"]].append((point["round"], point["ub"]))

    lb_rounds_new = list(map(int, lowerbounds.keys()))
    lb_values_new = list(lowerbounds.values())

    plt.figure(figsize=(12, 3))

    # 绘制UB数据点和连接线
    for i, pname in enumerate(pnames):
        alpha=0.5
        is_integer = ub_data[i].get("allInteger", False)
        if is_integer:
            color = 'red' 
            alpha = 1 
            s = 100
            print(pname)
        else:
            color = '#0072BD'
            s = children_counts[pname]*100+1
        if ub_data[i].get("fathomed", False) and not is_integer:
            color = 'orange' 
            s = 50
        round = brounds[i]
            # alpha = 0.5
        plt.scatter(round, ubs[i], color=color, alpha=alpha, s=s ,zorder = 3, edgecolor='none')

        # 连接前一个点，使用浅色线条
        if pre_pnames[i] is None:
            continue
        pre_index = pnames.index(pre_pnames[i])
        if ubs[pre_index] is None:
            continue
        plt.plot([brounds[i], brounds[pre_index]], [ubs[i], ubs[pre_index]], 
                 'lightgrey', linestyle='--', linewidth = 1, zorder = 0)
        
    # for mb, points in points_by_mb.items():
    #     patch = smooth_convex_hull(np.array(points))
    #     if patch:
    #         plt.gca().add_patch(patch)


    # 绘制Lowerbounds的线段
    plt.plot(lb_rounds_new, lb_values_new, color='green', linewidth=2, label='lowerbound', zorder = 0)

    plts = [plt.Line2D([], [], color='green', linewidth=2, label='lowerbound'),
        plt.Line2D([],[],color="#0072BD", linestyle='None',marker = 'o',  label="main chain"),
        plt.Line2D([],[],color="red", linestyle='None',marker = 'o', label="integer"),
        plt.Line2D([],[],color="orange", linestyle='None',marker = 'o',  label="fathomed"),]
        # plt.Line2D([100],[100],color="black", linestyle='None',marker = 'o', label="fork"),
        # plt.Line2D([100],[100],color="#9acd32", linestyle='None',marker = 'o', label="unpublished")]
    
    plt.xlabel('Round')
    plt.ylabel('Values')
    # plt.title('Scatter Plot with UB and Lowerbounds')
    plt.ylim(-25, -5) 
    plt.xlim(-1, 30) 
    plt.grid(True)
    plt.legend(handles = plts, loc = "lower right")
    strategy = " "
    if data['openblk_st'] == bb.OB_RAND and data['openprblm_st'] == bb.OP_BEST:
        strategy = 'BFS'
    if data['openblk_st'] == bb.OB_DEEP and data['openprblm_st'] == bb.OP_RAND:
        strategy = 'DFS'
    if data['openblk_st'] == bb.OB_BREATH and data['openprblm_st'] == bb.OP_RAND:
        strategy = 'BrFS'
    if data['openblk_st'] == bb.OB_RAND and data['openprblm_st'] == bb.OP_RAND:
        strategy = 'Rand'
    plt.savefig(SAVE_PREFIX + f"\\boundsv{data.get('var_num')}m{data.get('miner_num')}{strategy}_{time.strftime('%H%M%S')}.svg")
    plt.show()

def plot_bounds_fig3(data_list:list[dict], type):
    # 读取并解析文件
    # with open(file_path_new, 'r') as file:
    #     json_data_new = json.load(file)
    # 开始绘图
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 16  # 调整字号大小

    def lighten_color(color, amount=0.6):
        c = to_rgb(color)
        c = [1 - (1 - x) * amount for x in c]
        return c

    def get_pname(pname_list:list):
        if pname_list == "None":
            return None
        return tuple(tuple(x) if isinstance(x, list) else x for x in pname_list)

    def count_children(ub_data):
        # 更新每个点的后续点数量
        for point in ub_data:
            pre_pname = get_pname(point["pre_pname"])
            # 遍历所有点，增加前序点的后续点计数
            while pre_pname and  pre_pname != ((0,0),0):
                children_counts[pre_pname] += 1
                # 找到前序点的前序点
                pre_point = point_lookup.get(pre_pname)
                if pre_point is None:
                    break
                pre_pname = get_pname(pre_point["pre_pname"])
        print("count chilren finished")

    def draw_int_path(path, color = "#FF8283", linewidth=1.5, linestyle='--'):
        for i in range(len(path) - 1):
            start_point = (path[i]['bround'], path[i]['ub'])
            end_point = (path[i + 1]['bround'], path[i + 1]['ub'])
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                    color = color, linestyle=linestyle, linewidth=linewidth, alpha=0.7, zorder=6)


    # def get_pre_root(point):
    #     if point['pre_pname'] == None:
    #         return None
    #     pre_point_rows = ub_df[ub_df['pname'] == point['pre_pname']]
    #     pre_point = pre_point_rows.iloc[0] if not pre_point_rows.empty else None
    #     while pre_point is not None and pre_point['block'] == point['block']:
    #         point = pre_point
    #         pre_point_rows = ub_df[ub_df['pname'] == point['pre_pname']]
    #         pre_point = pre_point_rows.iloc[0] if not pre_point_rows.empty else None
    #     if pre_point is None:
    #         return None 
    #     return root_df[root_df['block'] == pre_point['block']].iloc[0]

    
    data = data_list[0]
    ub_data = data.get("ubdata", [])
    columns = ["miner", "block", "round", "bround", "pname", "pre_pname", "ub", "fathomed", "allInteger", "isFork", "isKeyblock"]
    ub_data = [dict(zip(columns, point)) for point in ub_data]
    if type != MAXSAT:
        for point in ub_data:
            if point["ub"] is None:
                continue
            point["ub"] = point["ub"]
    lowerbounds = data.get("lowerbounds", {})
    point_lookup = {get_pname(point['pname']): point for point in ub_data}
    children_counts = defaultdict(lambda : 1) 
    count_children(ub_data)
    
    # print(children_counts)

    # 创建数据帧
    ub_df = pd.DataFrame(ub_data)
    # ub_df['ub'] = -ub_df['ub']
    ub_df['children_count'] = ub_df['pname'].apply(lambda x: children_counts[get_pname(x)])
    ub_df['pname'] = ub_df['pname'].apply(lambda x: get_pname(x))
    ub_df['pre_pname'] = ub_df['pre_pname'].apply(lambda x: get_pname(x) if x != "None" else None)
    ub_df['is_main'] = ~ub_df['isFork'] & ~ub_df['fathomed'] & (ub_df['block'] != "None")
    ub_df['is_fathomed'] = ub_df['fathomed']
    # sampled_df = ub_df.sample(frac=0.01, random_state=0)  # 调整抽样比例
    # root_df = ub_df.copy().groupby('block').apply(lambda x: x.nsmallest(1, 'round')).reset_index(1,drop=True)
    # main_df = ub_df[ub_df["fathomed"] == False].copy()
    # min_ub_df = main_df[main_df['ub'] == main_df.groupby('block')['ub'].transform('min')]
    # max_ub_value = main_df.groupby('block')['ub'].max()
    # block_df = pd.merge(min_ub_df, max_ub_value, on='block', suffixes=('_min', '_max'))
    # # block_df = main_df.groupby('block').agg(min_ub=('ub', 'min'), max_ub=('ub', 'max'), bround=('bround', 'min')).reset_index()
    # pre_rows = []
    # for _, row in root_df.iterrows():
    #     pre_pname = row['pre_pname']
    #     matched_row = ub_df[ub_df['pname'] == pre_pname]
    #     pre_rows.append(matched_row)
    # pre_df = pd.concat(pre_rows)
    # bpre_rows = []
    # for _, row in block_df.iterrows():
    #     pre_pname = row['pre_pname']
    #     matched_row = ub_df[ub_df['pname'] == pre_pname]
    #     bpre_rows.append(matched_row)
    # print("plot")
        
    # bpre_df = pd.concat( bpre_rows)
    # children_counts[((0,0),0)]=bpre_df['children_count'].max()
    # print(ub_df)
    # print(block_df)
    # print(root_df)
    # print(pre_df)

     # sns.set(style="white")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    # 标记allInteger路径
    def get_draw_all_integer_path():
        int_paths = []
        intpath_points = set()
        max_bround_point = None
        max_bround = float('-inf')
        for point in ub_data:
            if point['allInteger'] and point["block"]!= "None":
                cur_point = point
                path = [cur_point]
                if point['bround'] > max_bround:
                    max_bround = point['bround']
                    max_bround_point = point
                if point['bround'] == max_bround and point['ub'] < max_bround_point['ub']:
                    max_bround = point['bround']
                    max_bround_point = point
                while cur_point['pre_pname'] != 'None':
                    pre_pname = get_pname(cur_point['pre_pname'])
                    pre_point = point_lookup.get(pre_pname)
                    if pre_point:
                        path.append(pre_point)
                        intpath_points.add(get_pname(point['pname']))
                        cur_point = pre_point
                    else:
                        break
                int_paths.append(path)
        # for path in int_paths:
        #     if max_bround_point in path:
        #         # 用红色标记这条路径
        #         draw_int_path(path, "red", 5,'-')
        #     else:
        #         # 其他路径使用默认颜色
        #         draw_int_path(path)
        # sampled_points = set(root_df['pname']).union(intpath_points).union((((0,0),0)))
    
    # get_draw_all_integer_path()

    
    # sampled_points = set(ub_df.sample(frac=0.1, random_state=0)['pname'])
    # sampled_points = sampled_points.union(intpath_points)
    # ["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"] 
    # ["#FF8283", "#0D898A","#f9cc52","#5494CE", ] '#00796B' '#ff8c00' '#b22222'
    # 绘制UB数据点和连接线
    smain = 10
    s = 5 if type == TSP else 80
    sopt = 50
    rasterized=False if type == MIPLTP else True
    if type != TSP:
        sns.scatterplot(x="bround",y ="ub",
                        data = ub_df[(ub_df["fathomed"] == False) & (ub_df["block"]!= "None")] ,
                        s = smain, color = '#0072BD', rasterized=rasterized ,edgecolor="none",
                        zorder = 5, alpha = 0.7) ,
    
    print("plot0")
    sns.scatterplot(x="round",y ="ub",
                    data = ub_df[(ub_df["fathomed"] == False) & (ub_df["block"]!= "None")] , 
                    s = smain, color = '#0072BD', rasterized=rasterized ,edgecolor="none",
                    zorder = 5, alpha = 0.7)
    print("plot1")
    sns.scatterplot(x="round",y ="ub",
                    data = ub_df[(ub_df["fathomed"]== True) & 
                                 (ub_df["allInteger"]==False) & (ub_df["block"]!= "None")] , 
                    color = "#ff8c00", s= s,rasterized=rasterized,edgecolor="none",alpha = 0.5)
    print("plot2")
    sns.scatterplot(x="round",y ="ub",data = ub_df[(ub_df["block"]== "None")], 
                    color = "#9acd32", s= s,rasterized=rasterized,edgecolor="none",zorder = 4, alpha = 0.5)
    print("plot3")
    # sns.scatterplot(x="bround",y ="ub",data = ub_df[(ub_df["isFork"]== True) & (ub_df["block"]!= "None")] , 
    #                 color = "black", s= s,rasterized=rasterized,edgecolor="none",zorder = 4, alpha = 0.5)
    print("plot4")
    sns.scatterplot(x="bround",y ="ub",data = ub_df[(ub_df["allInteger"] == True) & (ub_df["block"]!= "None")] , 
                    color = "#BEA9E9", s = 30 ,rasterized=rasterized,edgecolor="none",zorder = 6) 
    sns.scatterplot(x="round",y ="ub",data = ub_df[(ub_df["isKeyblock"]== True)], 
                    color = "r",  s= 100,rasterized=rasterized,edgecolor="none",zorder = 6)
    print("plot scatter finishhed")
    # point_norm = mcolors.Normalize(vmin=0, vmax=bpre_df['children_count'].max())
    # print(bpre_df['children_count'].max())
    blues = plt.cm.Blues
    # my_blues = mcolors.LinearSegmentedColormap.from_list("my_blues", 
    #["#caf0f8","#caf0f8","#ade8f4","#90e0ef","#48cae4","#00b4d8","#0096c7", "#0077b6","#003049"])#"#caf0f8" ,
    drawRect = True if type != TSP else False
    def adjust_width_for_log_scale(center_x, desired_width, base=10):
        # Calculate the factor to adjust the width in log scale
        factor = (np.log10(center_x + desired_width/2) - np.log10(center_x - desired_width/2)) / desired_width
        
        # Adjust the width and calculate the new left edge
        adjusted_width = desired_width / factor
        left = center_x - adjusted_width / 2
        
        return left, adjusted_width
    
    # if drawRect:
    #     blocks = []
    #     i=0
    #     for _, row in block_df.iterrows():
    #         if row['bround'] == -1:
    #             continue
    #         if row['block'] in blocks:
    #             continue
    #         i+=1
    #         print(i)
    #         print(row)
    #     # 提取每个block的数据
    #         blocks.append(row['block'])
    #         min_ub = row['ub_min']
    #         max_ub = row['ub_max']
    #         # 绘制圆角矩形
    #         width = 0.04 if type == MAXSAT else 0.06
    #         children_count = children_counts[row['pre_pname']]
    #         color = blues(1- point_norm(children_count))
    #         left, width = adjust_width_for_log_scale(row['bround'], width)
    #         # 计算对数尺度下的矩形边界
    #         rect = Rectangle((left, min_ub), width, max_ub - min_ub, 
    #                         linewidth=1.5, edgecolor='#5494CE',  facecolor = color,#facecolor='#00b4d8',
    #                         linestyle='-', capstyle='round', joinstyle='round',
    #                         rotation_point='center',alpha = 0.4, zorder = 2)
    #         # rect = FancyBboxPatch((row['bround'] - 500, min_ub), 
    #         #                   1000, max_ub - min_ub, 
    #         #                   boxstyle="round,pad=0.2,rounding_size=100", linewidth=1, 
    #         #                   edgecolor='none', facecolor='#0096c7', linestyle='--')
    #         plt.gca().add_patch(rect)
    #     for idx, block_row in block_df.iterrows():
    #         # 获取block点的坐标
    #         block_round = block_row['bround']
    #         block_ub = block_row['ub_min']

    #         # 获取pre点的坐标
    #         pre_row = pre_df[pre_df['pname'] == block_row['pre_pname']]
    #         if not pre_row.empty:
    #             pre_round = pre_row.iloc[0]['bround']
    #             pre_ub = pre_row.iloc[0]['ub']
    #             # 绘制折线连接两点
    #             plt.plot([pre_round, block_round, block_round], [pre_ub, pre_ub, block_ub], color = '#5494CE',
    #                     linestyle='--',linewidth = 1.5,alpha = 0.5, zorder = 0)#color='#CFCFCF'

    # my_oranges = mcolors.LinearSegmentedColormap.from_list("my_oranges", ["white", "#ee9b00"])
    
    # sns.kdeplot(data=ub_df[ub_df['is_main']],     x='round', y='ub', cmap="Blues" , fill=True, bw_adjust=0.2, zorder = 0)
    # sns.kdeplot(data=ub_df[ub_df['is_fathomed']], x='bround', y='ub', cmap="Oranges", fill=True, bw_adjust=0.2, zorder = 0)
    # sns.kdeplot(data=ub_df[ub_df['is_fathomed']], x='bround', y='ub', cmap="Oranges", fill=True, bw_adjust=0.1, zorder = 0)
    point_norm = mcolors.Normalize(vmin=0, vmax=math.log(max(children_counts.values())))
    
    point_cmap = plt.cm.Blues

    def draw_point(point,pre_point):
        # print(point["block"],point["pname"],point["bround"])
        alpha=0.6
        color = '#0072BD'  # 默认颜色
        if math.log(point['children_count']) > 10:
            zorder = 5
        elif math.ceil(math.log(point['children_count'])) > 5:
            zorder = 4
        elif math.ceil(math.log(point['children_count'])) > 3:
            zorder = 3
        elif math.ceil(math.log(point['children_count'])) > 1:
            zorder = 2
        elif math.ceil(math.log(point['children_count'])) >= 0:
            zorder = 1  
        s = ((point['children_count'])**0.5)*30+10
        # s = 10
        # print(math.log(point['children_count']))
        color = point_cmap(1- point_norm(math.log(point['children_count'])))
        # color = 
        if pre_point["allInteger"]:
            color = 'r' 
            alpha = 1 
            s = 30
            zorder = 50
            print(point["pname"])
        if pre_point["fathomed"] and not point["allInteger"]:
            color = 'orange' 
            s = 1
        if pre_point["isFork"]:
            color = "black"
            alpha = 1
            s = 50
        if pre_point["block"] == "None":
            color = "#9acd32"
            alpha = 0.5
            s = 50
            # alpha = 0.5
        plt.scatter(point["bround"], pre_point["ub"], color=color, alpha=alpha, 
                    s =s ,zorder =  zorder, rasterized=rasterized,
                    edgecolor='none')
    
    def get_pre_point(point):
        pre_point_rows = ub_df[ub_df['pname'] == point['pre_pname']]
        if pre_point_rows.empty:
            print("not foundd pre", point['pname'])
            return point
        return pre_point_rows.iloc[0]       
    
    # if type == TSP:
    #     for i, point in ub_df.iterrows():
    #         if point['pname'] not in sampled_points:
    #             continue  # 只绘制抽样的点
    #     #     # if point['block'] != "B1":
    #     #     #     continue
    #     #     # print(point['pname'])
    #     #     # if not point["allInteger"]:
    #     #     #     continue
    #     #     # if point['pname'] in sampled_points or (point["fathomed"] and not point["allInteger"]):
    #         pre_point = get_pre_point(point) if not point["allInteger"] else point
    #         draw_point(point, pre_point)

    # # 绘制Lowerbounds的线段
    # # 处理Lowerbounds数据
    lb_rounds_new = list(map(int, lowerbounds.keys()))
    if type == MAXSAT:
        lb_values_new = [v for v in lowerbounds.values()]
    else:
        lb_values_new = [v for v in lowerbounds.values()]
    plt.plot(lb_rounds_new, lb_values_new, color="#66A266", linewidth=8, zorder = 3)

    plts = [plt.Line2D([], [], color='green', linewidth=2, label='upper bounds'),
        plt.Line2D([],[],color="red", linestyle='None',marker = 'o', label="integer"),
        plt.Line2D([],[],color="#0072BD", linestyle='None',marker = 'o',  label="main-chain"),
        plt.Line2D([],[],color="orange", linestyle='None',marker = 'o',  label="fathomed"),
        plt.Line2D([],[],color="black", linestyle='None',marker = 'o', label="fork"),
        plt.Line2D([],[],color="#9acd32", linestyle='None',marker = 'o', label="unpublished")]
    
    ax.axhline(y=4000, color='r', linestyle='--',linewidth = 1, zorder = 3)
    ax.axhline(y=3500, color='r', linestyle='--',linewidth = 1, zorder = 3)
    # ax.axhline(y=3450, color='r', linestyle='--',linewidth = 1, zorder = 3)
    # ax.axhline(y=3400, color='r', linestyle='--',linewidth = 1, zorder = 3)
    # ax.axhline(y=3330, color='r', linestyle='--',linewidth = 1, zorder = 3)
    
    plt.xlabel('Round')
    plt.ylabel('Values')
    # plt.xscale("log")
    
    fig.subplots_adjust(left=0.079, bottom=0.1, right=0.98, top=0.98)
    if type == TSP:
        plt.xlim(100, 200000)
        plt.ylim(2500, 4000)
    elif type == MAXSAT:
        plt.xlim(100, 2500)
        plt.ylim(59, 63)
    elif type == MIPLTP:
        plt.xlim(10, 430)
        plt.ylim(180, 250)
    # plt.ylim(180, 250)
    # plt.xlim(-30, 430)
    # plt.title('Scatter Plot with UB and Lowerbounds')
    # plt.grid(True)
    # plt.legend(handles = plts)
    # ax.set_rasterized(True)
    plt.savefig(SAVE_PREFIX + f"\\bounds_maxsat{time.strftime('%H%M%S')}.svg", dpi=300)
    # plt.show()

def plot_solveround_workload_fig4(df:pd.DataFrame):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times new roman']
    plt.rcParams['font.size'] = 12
    colors = ["#FF8283", "#0D898A","#f9cc52","#5494CE", ]
    # sns.set(style="whitegrid")
    def plot_solve_rounds(ax:plt.Axes, df_med:pd.DataFrame, df_easy:pd.DataFrame):
        df_med.groupby('miner_num')['ave_solve_round']
        ax.plot(df_med['miner_num'], df_med['ave_solve_round'], label='medium', marker='o',color =colors[1])
        ax.plot(df_easy['miner_num'], df_easy['ave_solve_round'], label='easy', marker='x',color =colors[2])
        ax.set_xlabel('Number of miners')
        ax.set_ylabel('Solving round')
        ax.legend()
    
    def plot_speedup(ax:plt.Axes, df_med:pd.DataFrame, df_easy:pd.DataFrame, 
                     m1_med:pd.DataFrame, m1_easy:pd.DataFrame):
        speedup_100 = m1_med / df_med['ave_solve_round']
        speedup_50 = m1_easy / df_easy['ave_solve_round']
        ax.plot(df_med['miner_num'], speedup_100, marker='o',color =colors[1])
        ax.plot(df_easy['miner_num'], speedup_50, marker='x',color =colors[2])
        # ax2 = ax.twinx()
        # ax2.plot(df_100['miner_num'], speedup_100/df_100['miner_num'])
        # ax2.plot(df_50['miner_num'], speedup_100/df_100['miner_num'])
        # ax.set_xlabel('Number of miners')
        ax.set_xlabel(' ')
        ax.set_ylabel('Speed up',labelpad = 12)
        # ax.legend()
    def plot_efficiency(ax:plt.Axes, df_med:pd.DataFrame, df_easy:pd.DataFrame, 
                       m1_med:pd.DataFrame, m1_easy:pd.DataFrame):
        speedup_100 = m1_med / df_med['ave_solve_round'] / df_med['miner_num']
        speedup_50 = m1_easy / df_easy['ave_solve_round']/ df_med['miner_num']
        ax.plot(df_med['miner_num'], speedup_100, marker='o',color =colors[1])
        ax.plot(df_easy['miner_num'], speedup_50, marker='x',color =colors[2])
        # ax2 = ax.twinx()
        # ax2.plot(df_100['miner_num'], speedup_100/df_100['miner_num'])
        # ax2.plot(df_50['miner_num'], speedup_100/df_100['miner_num'])
        ax.set_xlabel('Number of miners')
        ax.set_ylabel('Efficiency')
    
    def plot_workload(ax:plt.Axes, df_med:pd.DataFrame):
    # Extracting the required metrics
        metrics = df_med[['miner_num', 'main', 'fork', 'unpub']]
        metrics = metrics.set_index('miner_num')
        
        colors=["#1b4332","#40916c","#74c69d","#95d5b2"]
        # colors=["#31572c", "#4f772d", "#90a955","#ecf39e"] "#2d6a4f","#52b788"
        width = 1.5
        ax.bar(df_med['miner_num'], df_med['main'], 
               label='main', color=colors[0], alpha=0.8,width=width,zorder=1)
        ax.bar(df_med['miner_num'], df_med['fork'], bottom=df_med['main'], 
               label='fork', color=colors[1], alpha=0.8,width=width,zorder=1)
        ax.bar(df_med['miner_num'], df_med['unpub'], bottom=df_med['main'] + df_med['fork'], 
               label='unpublished', color=colors[2], alpha=0.8,width=width,zorder=1)
        # metrics.plot(kind='bar', ax=ax, stacked=True, color=colors,)
        # Plotting
        # metrics.plot(kind='bar', ax=ax, stacked=True)
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5,zorder=0)
        # ax.set_xlabel('Number of miners')
        ax.set_xlabel(' ')
        ax.set_ylabel('Workload')
        ax.legend()

    def plot_workload_per_miner(ax:plt.Axes, data):
        # Extracting the adjusted metrics
        metrics_per = data[['miner_num', 'main_per', 'fork_per', 'unpub_per']]
        metrics_per = metrics_per.set_index('miner_num')
        colors=["#1b4332","#40916c","#74c69d","#95d5b2"]
        # colors=["#31572c", "#4f772d", "#90a955","#ecf39e"] "#2d6a4f","#52b788"
        width = 1.5
        ax.bar(df_med['miner_num'], df_med['main_per'], 
               label='main', color=colors[0], alpha=0.8,width=width,zorder=1)
        ax.bar(df_med['miner_num'], df_med['fork_per'], bottom=df_med['main_per'], 
               label='fork', color=colors[1], alpha=0.8,width=width,zorder=1)
        ax.bar(df_med['miner_num'], df_med['unpub_per'], bottom=df_med['main_per'] + df_med['fork_per'], 
               label='unpublished', color=colors[2], alpha=0.8,width=width,zorder=1)
        # Plotting
        # metrics_per.plot(kind='bar', ax=ax, stacked=True)
        ax.set_xlabel('Number of miners')
        ax.set_ylabel('Workload per miner',labelpad = 12)
        ax.legend()

    def plot_workload_decrease(ax:plt.Axes, data):
        # Calculate metrics for miner_num = 1 as baseline
        m1_df = data[data['miner_num'] == 1][['main_per', 'chain_per', 'total_per']].iloc[0]
        print(m1_df)
        metrics = ['main_per', 'chain_per', 'total_per']
        labels = ['main', 'chain', 'total']
        for i, (metric,label) in  enumerate(zip(metrics,labels)):
            data[f'{metric}_decrease'] = data[metric] / m1_df[metric]
        # for d in data.iterrows():
            ax.plot(data['miner_num'], data[f'{metric}_decrease'], label=label, marker='o',color = colors[i])
        print(data)
        ax.set_xlabel('Number of miners')
        ax.set_ylabel('Decrease ratio of \nworkload per miner')
        ax.legend()

    def plot_fork_rate(ax:plt.Axes, data:pd.DataFrame):
    # Filter data by var_num
        difficulties = data['difficulty'].unique()
        for i, difficulty in enumerate(sorted(difficulties)):
            df_filtered = data[data['difficulty'] == difficulty]
            grouped = df_filtered.groupby('miner_num')['ave_mb_forkrate'].mean().reset_index()
            ax.plot(grouped['miner_num'], grouped['ave_mb_forkrate'], label=f'{difficulty}', marker='o',color = colors[i])
        # ax.set_xlabel('Number of miners')
        ax.set_xlabel(' ')
        ax.set_ylabel('Fork rate of mini-blocks')
        ax.legend(title = "difficulty")
    
    fig = plt.figure(figsize=(10, 12))
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.5, 1.5],width_ratios=[1.2,1])
    axSolveRounds = fig.add_subplot(grid[0:2, 0]) # ax_a will span two rows.
    axSpeed1 = fig.add_subplot(grid[0, 1])
    axSpeed2 = fig.add_subplot(grid[1, 1])
    axWork = fig.add_subplot(grid[2, 0]) # Placeholder for ax_d (e in the description)
    axWorkPer = fig.add_subplot(grid[3, 0])
    axFork = fig.add_subplot(grid[2, 1])
    axWorkDec = fig.add_subplot(grid[3, 1])
    ax_list = [axSolveRounds, axSpeed1,axSpeed2, axWork, axWorkPer, axFork, axWorkDec]
    for ax in ax_list:
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 设置为浅灰色
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5,zorder=0)

    df['main'] = df['ave_acp_subpair_num']
    df['fork'] = df['ave_subpair_num'] - df['ave_acp_subpair_num']
    df['unpub'] = df['ave_subpair_unpubs'] - df['ave_subpair_num']
    df['main_per'] = df['main'] / df['miner_num']
    df['fork_per'] = df['fork'] / df['miner_num']
    df['unpub_per'] = df['unpub'] / df['miner_num']
    df['chain_per'] = df['ave_subpair_num'] / df['miner_num']
    df['total_per'] = df['ave_subpair_unpubs'] / df['miner_num']
    df_med = df[(df['difficulty'] == 5) & (df['var_num'] == 100)]
    df_easy = df[(df['difficulty'] == 5) & (df['var_num'] == 50)]
    sr_med = df_med.groupby('miner_num')['ave_solve_round'].mean().reset_index()
    sr_easy = df_easy.groupby('miner_num')['ave_solve_round'].mean().reset_index()
    m1sr_med = df_med[df_med['miner_num'] == 1]['ave_solve_round'].mean()
    m1sr_easy = df_easy[df_easy['miner_num'] == 1]['ave_solve_round'].mean()
    plot_solve_rounds(axSolveRounds, sr_med, sr_easy)
    plot_speedup(axSpeed1, sr_med, sr_easy, m1sr_med, m1sr_easy)
    plot_efficiency(axSpeed2, sr_med, sr_easy, m1sr_med, m1sr_easy)
    plot_workload(axWork, df_med)
    plot_workload_per_miner(axWorkPer, df_med)
    plot_fork_rate(axFork,df[df['var_num'] == 100])
    plot_workload_decrease(axWorkDec,df_med)
    fig.subplots_adjust(left=0.079, bottom=0.1, right=0.98, top=0.98,hspace=0.4)

    
    
    plt.show()

def plot_mbtime_grow_fig5(data_df:pd.DataFrame=None):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14
    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.5, 1.5, 2],width_ratios=[1.5, 1, 1.5,1]) # 3 rows, 2 columns
    # sns.set(style="whitegrid")
    # colors = ["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"]
    colors = ["#FF8283", "#0D898A","#f9cc52","#5494CE", ]

    def plot_means(ax:plt.Axes, data_df):
        # Unique miners for different lines
        
        miners = data_df['miner_num'].unique()
        print(miners)
        for i , miner in enumerate(miners):
            miner_data = data_df[data_df['miner_num'] == miner]
            # ax.errorbar(miner_data['difficulty'], miner_data['mean'], yerr=miner_data['std'], label=miner)
            ax.plot(miner_data['difficulty'], miner_data['mean'],  label=miner, marker = 'o',color = colors[i])

        # ax.set_title(title)
        ax.set_xlabel('Difficulty')
        ax.set_ylabel('Means of block times')
        ax.legend(title="miner num",fontsize = 12)
        ax.set_xticks([3,5,7,9])
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5)
    
    def plot_violin(ax:plt.Axes, data_df:pd.DataFrame):
        # colors=["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"]
        sns.violinplot(x='mb_times',y='miner_num', hue='difficulty', inner="box", 
                       data=data_df, ax=ax, orient='h',palette = colors[:4], linewidth = 0.1,alpha = 0.1)
        # sns.pointplot(x='mb_times', y='difficulty', data=data_df, hue='miner_num',
            #   orient='h', join=False, markers='D', palette='dark', ci=None, estimator='mean')
        ax.set_xlim([0,150])
        # ax.set_ylabel('Difficulty', labelpad = -10, loc='top')
        leg = ax.legend(title="difficulty",fontsize = 12,loc = "lower right", bbox_to_anchor=(1, 0.05))
        leg.get_title().set_fontsize('12')
        ax.set_ylabel('The number of miners')
        ax.set_xlabel('Block times')

    def plot_growthrate(ax:plt.Axes, data_df:pd.DataFrame):
        miners = data_df['miner_num'].unique()
        for i, miner in enumerate(miners):
            miner_data = data_df[data_df['miner_num'] == miner]
            # ax.errorbar(miner_data['difficulty'], miner_data['mean'], yerr=miner_data['std'], label=miner)
            ax.plot(miner_data['difficulty'], 1/miner_data['ave_mb_growth'],  label=miner, marker = 'o',color = colors[i])

        # ax.set_title(title)
        ax.set_xlabel('Difficulty')
        ax.set_ylabel('Growth rate')
        ax.legend(title="miner num",fontsize = 12)
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5)
        # sns.boxplot(x='difficulty', y='grow_proc', data=data_df, ax=ax)
    
    def plot_grow_proc(ax:plt.Axes, data_df:pd.DataFrame):
        path = pathlib.Path(".\Results\\20240206\\224045\m5d5vmaxsatevaluation results.json")
        with open(path,'r') as file:
            new_data = json.load(file)
            # print(dd)
            # new_data = pd.DataFrame([dd])
            # print(new_data)
            ax.plot(range(len(list(new_data['mb_times']))), list([d for d in new_data['mb_times']]), 
                    label = "block times",alpha=0.8,color="#f9cc52")
            ax.hlines(np.sum(new_data['mb_times'])/len(list(new_data['mb_times'])), 
                      0, len(list(new_data['mb_times'])),linestyles='--',linewidth = 2, color="orange",label = "mean")
        # sns.lineplot()
        # c = plt.cm.Greens
        # # point_norm = mcolors.Normalize(vmin=0, vmax=len(data_df['miner_num'].unique()))
        # i=1
        # data_df.copy().reset_index()
        # c = ["#95d5b2", "#74c69d", "#52b788","#40916c"]
        # for i,data in data_df.iterrows():
        #     print(data_df)
        #     # ax.hlines(1/data['ave_mb_growth'], 0, 5000,color=c[i-4],linestyles='--',linewidth = 1)
        #     # data = data_df[data_df['difficulty'] == d]
        #     # ax.plot(range(len(list(data['grow_proc']))), list([1/d for d in data['grow_proc']]), 
        #     #         label = data['difficulty'],c=c[i-4])
        #     # sns.scatterplot(data=data,y='mb_times',s=1,alpha=0.1,rasterized=True)
        #     ax.scatter(range(len(list(data['mb_times']))), list([d for d in data['mb_times']]), 
        #             label = data['difficulty'],s=2,alpha=0.5,rasterized=True)
        #     ax.hlines(np.sum(data['mb_times'])/len(list(data['mb_times'])), 
        #               0, 5000,linestyles='--',linewidth = 1, color="orange")
            # i+=1
        # ax.set_ylim([0,50])
        # ax.set_xlim([-30,3000])
        ax.legend(fontsize = 12)
        ax.set_xlabel('Blocks')
        ax.set_ylabel('Block time')
        # ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5)
    data_list = []
    with open(pathlib.Path("Result_Data\\0207tspm125mbtimes.json"), 'r') as f:
        jsondata_list = f.read().split('\n')[:-1]
        for jsondata in jsondata_list:
            data_list.append(json.loads(jsondata))
        data_df = pd.DataFrame(data_list)
        data_df = data_df.sort_values(by=["miner_num","difficulty" ])

    mb_times_df = data_df.copy().explode('mb_times')
    mb_times_df['mb_times'] = pd.to_numeric(mb_times_df['mb_times'])
    grow_df = data_df.copy().explode('grow_proc')
    grow_df['grow_proc'] = pd.to_numeric(grow_df['grow_proc'])
    # exploded_df = pd.to_numeric(exploded_df['mb_times'])
    # grouped_a = exploded_df.groupby(['miner_num', 'difficulty'])['mb_times'].agg(['mean', 'std']).reset_index()
    data_df['mean'] = data_df['mb_times'].apply(lambda x: np.mean(x) if isinstance(x, list) else x)
    data_df['std'] = data_df['mb_times'].apply(lambda x: np.std(x) if isinstance(x, list) else x)
    print(data_df,mb_times_df)
    # data_df = data_df.groupby(['miner_num', 'difficulty'])['mb_times'].agg(['mean', 'std']).reset_index()
    # Create the subplots
    
    axMeans = fig.add_subplot(grid[0:2, 0]) # ax_a will span two rows.
    axTimesM1 = fig.add_subplot(grid[0, 1:3])
    axTimesM3 = fig.add_subplot(grid[1, 1:3])
    axGrowProc = fig.add_subplot(grid[2, 0:2]) # Placeholder for ax_d (e in the description)
    axGrowth = fig.add_subplot(grid[2, 2:4])
    axViolin = fig.add_subplot(grid[0:2, 3])

    plot_means(axMeans, data_df)
    plot_violin(axViolin, mb_times_df)
    plot_growthrate(axGrowth, data_df)
    plot_block_time_fig5(axTimesM1, data_df[data_df['miner_num'] == 1])
    plot_block_time_fig5(axTimesM3, data_df[data_df['miner_num'] == 2])
    plot_grow_proc(axGrowProc, data_df[(data_df['miner_num'] == 5) & (data_df['difficulty'].isin([5]))].copy())
    
    ax_list = [axMeans, axTimesM1, axTimesM3, axViolin,axGrowth, axGrowProc]
   
    
    for ax in ax_list:
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 设置为浅灰色
        # ax.text(-0.05, 1, label, transform=ax.transAxes, fontsize=16,
        #     verticalalignment='top', horizontalalignment='left',fontweight='bold')
        # # fig.text(0.1, 0.9, label, fontsize=16, fontweight='bold')
            
    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    # 假设这些是你想要的编号位置，根据你的图形尺寸进行适当调整
    label_positions = [(0.04, 0.97), (0.3, 0.97), (0.3, 0.67),(0.8, 0.97), 
                       (0.04, 0.39), (0.5, 0.39)]

    # 添加子图编号
    for pos, label in zip(label_positions, subplot_labels):
        fig.text(pos[0], pos[1], label, fontsize='large', fontweight='bold', 
                transform=fig.transFigure, ha='center', va='center')
    plt.tight_layout()

    if SAVE:
        plt.savefig(SAVE_PREFIX + "\\mbtimes.svg", dpi=220)
    plt.show()

def plot_block_time_fig5(ax:plt.Axes, data_df:pd.DataFrame):
    """"
    区块时间pdf，并标注峰值与均值
    """
    
    
    hist_list = []
    bins_list = [] 
    peaks_list = [] # 存储峰值
    prominences_list = [] # 峰值突出度
    avg_values = []  # 存储平均值
    mb_times_list = []
    difficulties=[]
    for _,data in data_df.iterrows():
        mb_times = np.array(data['mb_times'])
        mb_times_list.append(mb_times)
        difficulties.append(data["difficulty"])
        # 计算频率
        # hist, bins = np.histogram(
        #     data, bins=300, density=True)
        hist, bins = np.histogram(
            mb_times, bins=np.arange(min(mb_times), max(mb_times) + 2), density=True)
        hist_list.append(hist)
        bins_list.append(bins)
        # 找到直方图的峰值和突出度
        peaks, _ = find_peaks(hist)
        prominences = peak_prominences(hist, peaks)[0]
        prominence_threshold = 0.01
        peaks_list.append(peaks)
        prominences_list.append(prominences)
        # 计算平均值
        avg = np.mean(mb_times)
        avg_values.append(avg)
    colors = ['#1f77b4', '#32cd32', '#bcbd22', '#ffa500','#b22222',
            '#9467bd', '#87cefa', '#ff6347',]
    colors = ["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"]
    # colors = ["r","b","y","g","#BEA9E9"]
    # 绘图
    # fig, ax= plt.subplots(figsize=(10, 6.5))
    for plt_idx,(mb_times,hist,bins,peaks,prominences, avg,difficulty) in enumerate(
            zip(mb_times_list,hist_list, bins_list, peaks_list, prominences_list,avg_values,difficulties)):
        # ax.bar(bins[:-1], hist, width=1, align='edge', alpha=0.5, color=colors[plt_idx])
        ax.hist(mb_times, bins=np.arange(min(mb_times), max(mb_times) + 2), density=True, 
                alpha=0.3,histtype='stepfilled', edgecolor = "black",
                linewidth=0,color=colors[plt_idx],align="left",zorder = 4-plt_idx, label=difficulty)
        ax.plot(bins[:-1], hist, color=colors[plt_idx],linewidth = 1.5,zorder = 4)
        # 标记平均值位置
        avg_y = np.interp(avg, bins[:-1], hist)
        ax.axvline(x=avg, color=colors[plt_idx], linestyle='-',linewidth = 4,  alpha = 0.5)
        # ax.scatter(avg, avg_y, marker='o', color='red', zorder=5)
        ax.annotate(f'{avg:.2f}', 
                    weight='bold',
                    xy=(avg, avg_y), 
                    xytext=(plt_idx, plt_idx), 
                    textcoords='offset points',
                    color=colors[plt_idx])
    # # 添加 "Averages" 标签
    # ax.annotate('Means', 
    #             xy=(13, 0.06), 
    #             xytext=(0, 0), 
    #             textcoords='offset points',color='red',weight='bold',
    #             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', 
    #                       ec='k',lw=1))
    # ax.scatter(-100,-100,color="red",label="means",marker="o")
    # ax.set_xlabel(xlabel)
    ax.set_ylabel('PMF')
    x_tick_step = 5
    x_ticks = np.arange(0, max(mb_times) + x_tick_step, x_tick_step)
    x_ticks = np.insert(x_ticks, 0, min(mb_times))
    ax.set_xticks(x_ticks)
    # ax.set_xlim(min(data), max(data)+1)
    ax.set_xlim(min(mb_times), 50+1)
    ax.set_ylim(0,0.15)
    ax.legend(loc='best', title = "difficulty",fontsize = 12)



def plot_security_fig6():
    data_path = pathlib.Path(".\Result_Data\\1029attack_data2lite.json")
    data_list = []
    with open(data_path, 'r') as f:
        json_list = f.read().split('\n')[:-1]
        for json_data in json_list:
            data_list.append(json.loads(json_data))

    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 8
    df = pd.DataFrame([data for data in data_list if data["difficulty"] in [3,5,8,10]])
    df_sorted = df.sort_values(by=['safe_thre', 'difficulty'],ascending = [False,True])
    sorted_safe_thre = df_sorted['safe_thre'].unique()
    print(sorted_safe_thre)
    print(df)
    np.random.seed(0)  # 为了可重复性的示例
    sns.set(style="whitegrid", font='Times New Roman', font_scale=1)
    colors = ["#FF8283", "#0D898A","#f9cc52","#5494CE", ]

    # 创建图表
    heights = [3, 1, 1, 1]
    fig = plt.figure(figsize=(8, 7))
    # fig, (ax1, ax41, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 8), 
    #                                     gridspec_kw={'height_ratios': heights, 'hspace': 0})
    
    grid = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1],width_ratios=[1])
    ax1 = fig.add_subplot(grid[0, 0]) # ax_a will span two rows.
    ax41 = fig.add_subplot(grid[1, 0])
    # ax42 = fig.add_subplot(grid[1, 0])
    # ax43 = fig.add_subplot(grid[1, 2]) # Placeholder for ax_d (e in the description)
    ax2 = fig.add_subplot(grid[2, 0])
    ax3 = fig.add_subplot(grid[3, 0])

    ax1: plt.Axes
    ax41: plt.Axes
    ax42: plt.Axes 
    ax2: plt.Axes
    ax3: plt.Axes
    
    # 第一个图表：Rate的柱状图
    sns.barplot(x='safe_thre', y='ave_advrate', hue='difficulty', 
                palette = colors, data=df, ax=ax1, width = 0.7, order=sorted_safe_thre)
    ax1.set_xlabel(' ')  # 移除x轴标签，因为将与第二个图共享
    ax1.set_ylabel('Success probability',labelpad = 8)
    ax1.legend(title='Difficulty', loc = "upper left", bbox_to_anchor=(0.08, 0.98),fontsize=10)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xticklabels([])
    # ax1.grid(True)
    ax1.grid(axis='y')
    # ax1.grid(True)

    # 第二个图表：Prob的柱状图
    sns.barplot(x='safe_thre', y='ave_accept_advrate', hue='difficulty', 
                palette = colors, data=df, ax=ax2, width = 0.7,order=sorted_safe_thre)
    ax2.set_ylabel('Chain\nquality',labelpad = 11)
    ax2.set_xlabel('')  # 移除x轴标签，因为将与第二个图共享
    ax2.set_ylim(bottom=0 + 0.00001)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax2.get_legend().remove()
    ax2.set_xticklabels([])
    ax2.grid(True)
    ax2.grid(axis='y')
    ax2.invert_yaxis()
    # ax2.grid(True)

    # 调整子图间距并显示图表

    # 第三个图表：Wasted的箱型图
    
    wasted_path = pathlib.Path(".\Result_Data\\1029attack_data2lite.json")
    with open(wasted_path,'r') as f:
        data_list = []
        jsondata_list = f.read().split('\n')[:-1]
        for jsondata in jsondata_list:
            data_list.append(json.loads(jsondata))
        wasted_df = pd.DataFrame(data_list)
        print(wasted_df[wasted_df["difficulty"] == 5])
    sns.boxplot(x='safe_thre', y="ave_subpair_unpubs", data = wasted_df[wasted_df["difficulty"] == 5], 
                ax=ax3, width = 0.2, linewidth = 2,order=sorted_safe_thre)
    ax3.set_ylabel('Wasted\nworkload',labelpad = 22)
    ax3.set_xlabel('Safe threshold')
    ax3.grid(True)
    ax3.grid(axis='y')
    # ax3.invert_xaxis()
    # ax3.set_ylabel('Count')

    df.loc[df['difficulty'] == 5, 'safe_ratio'] = \
        df.loc[df['difficulty'] == 5, 'ave_advrate'] / df.loc[df['difficulty'] == 5, 'safe_thre']
    df_d5 = df[df["difficulty"] == 5].sort_values(by="safe_thre", ascending=False)
    # print(df_d5)
    # sns.lineplot(x='safe_thre', y='safe_thre', data=df_d5, ax=ax_inset,
    #          marker='x', linestyle="--", color="#0D898A", label="threshold",)
    # sns.lineplot(x='safe_thre', y='ave_advrate', data=df_d5, ax=ax_inset,
    #             marker='o', color="#0D898A", label="simulation",)
    # ax_inset.legend(loc = "upper left",bbox_to_anchor=(0.4, 1))
    # 
    # ax_inset2 = ax_inset.twinx()
    # sns.lineplot(x= "safe_thre",y = 'safe_ratio' , 
    #             data =df_d5,
    #             marker='o', color = "#BC5133")

    # ax_inset = ax1.inset_axes([0.45, 0.51, 0.5, 0.45])  # [x, y, width, height] in relative coordinates
    # ax41 = ax1.inset_axes([0.4, 0.5, 0.5, 0.4])
    original_xticks = [0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0003, 0.0001]

    # 视觉上均匀分布的x轴刻度位置
    visual_xticks = range(0, len(original_xticks))
    ax41.set_xticks(visual_xticks)
    ax41.set_xticklabels([])
    ax41.plot(visual_xticks, 'safe_thre' , 
                data = df[df["difficulty"] == 5].sort_values(by="safe_thre",ascending=False),  
                marker='x',linestyle = "--", color = "#0D898A",
                label = "threshlod")
    ax41.plot(visual_xticks, 'ave_advrate' , 
                data = df[df["difficulty"] == 5].sort_values(by="safe_thre",ascending=False),  
                marker='o',color = "#0D898A",
                label = "simulation")
    ax41.set_ylim(bottom=0, top=0.0051)
    # ax41.set_xticks([])
    # ax41.set_xticks(sorted_safe_thre)
    ax41.set_xlim(ax1.get_xlim())
    # print()
    # print(ax3.get_xlim())
    # ax41.invert_xaxis()
    ax41.legend(loc = "upper left",bbox_to_anchor=(0.3, 1.01),ncol=2)
    axins_2 = ax41.twinx()
    axins_2.plot(visual_xticks, 'safe_ratio' , 
                data = df[df["difficulty"] == 5].sort_values(by="safe_thre",ascending=False), 
                    marker='o', color = "#BC5133",alpha = 0.5)
    axins_2.spines['left'].set_color('#0D898A')  # Set the color of the y-axis to blue
    # axins_2.set_xticks(sorted_safe_thre)
    axins_2.set_xticks(visual_xticks)
    # axins_2.set_xticks([])
    axins_2.set_xticklabels([])
    # ax_inset.set_yticks([0.00])
    ax41.yaxis.label.set_color('#0D898A')
    ax41.tick_params(axis='y', colors='#0D898A')
    ax41.set_ylabel("Success\nprobability",labelpad=15)
    ax41.set_xlabel(" ")
    ax41.grid(axis='y')
    # labels = [item.get_text() for item in ax41.get_xticklabels()]
    ax41.tick_params(axis='x')
    # ax41.xaxis.set_major_locator(plt.MaxNLocator(5))
    axins_2.spines['right'].set_color('#BC5133')  # Set the color of the y-axis to blue
    axins_2.yaxis.label.set_color('#BC5133')
    axins_2.tick_params(axis='y', colors='#BC5133')
    axins_2.set_ylabel("Safety performance")
    # ax_inset.grid(False)
    axins_2.grid(False)

    axins1 = ax1.inset_axes([0.33, 0.4, 0.3, 0.5])
    axins2 = ax1.inset_axes([0.7, 0.4, 0.3, 0.5])

    # 调整子图间距
    # ax_inset = ax1.inset_axes([0.5, 0.3, 0.5, 0.65])
    # ax1.add_patch(Rectangle((5.6, 0.00001), 0.3, 0.0001, color='#45636A', fill=False, lw=1.5))
    # ax1.add_artist(ConnectionPatch(xyA=(5.9, 0.00011), xyB=(6.5, 0.000572), coordsA='data', coordsB='data',
    #                    axesA=ax1, axesB=ax1, color='#45636A',lw = 0.5))
    # ax1.add_artist(ConnectionPatch(xyA=(5.6, 0.00011), xyB=(3, 0.000572), coordsA='data', coordsB='data',
    #                     axesA=ax1, axesB=ax1, color='#45636A',lw = 0.5))
    
    atklog_path = pathlib.Path(".\Result_Data\simudata_collect084707.json")
    atklog_list = []
    with open(atklog_path, 'r') as f:
        json_list = f.read().split('\n')[:-1]
        for json_data in json_list:
            atklog_list.append(json.loads(json_data))
    atklog_df = pd.DataFrame(atklog_list)
    # print(atklog_df.loc[atklog_df['difficulty']==3]["atklog_mb"].iloc[0])
    plot_atklog_fig6(atklog_df.loc[atklog_df['difficulty']==3]["atklog_mb"].iloc[0], 
                     axins1, atklog_df.loc[atklog_df['difficulty']==3]["safe_thre"].iloc[0],
                     color = "#FF8283")
    plot_atklog_fig6(atklog_df.loc[atklog_df['difficulty']==7]["atklog_mb"].iloc[0], 
                    axins2, atklog_df.loc[atklog_df['difficulty']==7]["safe_thre"].iloc[0],
                    color = "#f9cc52")
    axins1.set_ylabel("Success probability", fontsize=10)
    axins1.tick_params(axis='x', labelsize=10)
    axins1.tick_params(axis='y', labelsize=10)
    axins1.set_xlabel("Blocks", fontsize=10)
    axins2.set_ylabel(" ",labelpad = 12)
    axins2.set_xlabel("Blocks", fontsize=10)
    axins2.tick_params(axis='y', labelsize=10)
    axins2.tick_params(axis='x', labelsize=10)
    axins1.legend(fontsize = 10)
    fig.subplots_adjust(left=0.13, bottom=0.08, right=0.9, top=0.98,hspace=0.05)
    plt.show()
    
def plot_atklog_fig6(atklog_mb:list, ax_inset:plt.Axes, safe_thre,color):
    """
    {"depth":0,"theory":0,"attack_num":0,"success_num":0,"success_rate":0}
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 24  # 调整字号大小
    
    atklog_mb = [atk for atk in atklog_mb if atk["success_num"] != 0 and atk["attack_num"] != 0]
    if len(atklog_mb) > 0:
        for atk in atklog_mb:
            success_rate = atk["success_num"] / atk["attack_num"]
            atk["success_rate"] = success_rate
        atklog_mb = sorted(atklog_mb, key=lambda x: x["theory"],reverse=True)
    attacks = range(len(atklog_mb))
    success_rates = [atk['success_rate'] if atk['success_rate']!=0 else 0 for atk in atklog_mb]
    theory_values = [atk['theory'] for atk in atklog_mb]
    lbs = [math.log(1/(50**atk['depth'])) for atk in atklog_mb]

    # fig = plt.figure(figsize=(10, 6.5))
    # ax_inset.bar(attacks, success_rates, label='simulation', color='orange',  width=1, alpha=0.7)
    ax_inset.fill_between(attacks, success_rates, color=color, alpha=0.5, label='simulation',edgecolor='none')
    ax_inset.plot(attacks, theory_values, label='theory ', color='#1f77b4', linestyle = "--",linewidth = 2)
    # plt.plot(attacks, lbs, label='lowerbound', color='green', alpha=0.7, linestyle = "--")
    # plt.axhline(y=math.log(1/(50**3.5)), label='Lowerbound', color="green", linestyle='--')
    ax_inset.axhline(y=safe_thre, label='threshold', color="red", linestyle='-.',linewidth = 2)
    # plt.ylim([-17.0,-6.0])
    ax_inset.set_xlim([0, len(atklog_mb)+10])
    ax_inset.set_yscale("log")
    ax_inset.set_xlabel('Blocks')
    ax_inset.set_ylabel('Success rate')
    # ax_inset.legend(loc = "lower left")
                    # , bbox_to_anchor=(1, 0.97))
    # plt.legend(loc = "best")
    ax_inset.grid()
    # ax.set_rasterized(True)
    # if SAVE:
    #     plt.savefig(SAVE_PREFIX + "\\atklogm10_001.svg", dpi=300)
    # plt.show()