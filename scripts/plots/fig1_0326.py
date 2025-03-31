import json
from pathlib import Path
import sys
import time
from collections import defaultdict

import pandas as pd
sys.path.append("E:\Files\gitspace\\bbb-github")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Ellipse,
    Rectangle,
)

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

def plot_feasible_regions_fig1(FIG_NUM):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 20  # 调整字号大小

    public = {'c' : np.array([1, -4]),
            'G_ub' : np.array([ [-1, 0], [0, -1], [-5, 10], [5, 10], [1,0]]),
            'h_ub' : np.array([0, 0, 11, 49, 5]),
            'x_lp': [3.8,  3],
            'z_lp': -8.2,
            }
    extra1 = {'G_ub': [[-1,0]],
            'h_ub': [-4],
            'x_lp': [4,  2.9],
            'z_lp': -7.6,}
    extra2 = {'G_ub': [[1,0]],
            'h_ub': [3],
            'x_lp': [3.0,  2.6],
            'z_lp': -7.4,}
    aux_line = None
    
    if FIG_NUM == 2:
        public['G_ub'] = np.append(public['G_ub'], [[1,0]], axis = 0)
        public['h_ub'] = np.append(public['h_ub'], [3], axis = 0)
        aux_line = extra1
        extra1 = {'G_ub': [[0, -1]],
            'h_ub': [-3],
            'x_lp': None,
            'z_lp': -10.089,}
        extra2 = {'G_ub': [[0, 1]],
            'h_ub': [2],
            'x_lp': [ 1.8,  2],
            'z_lp': -6.2,}
        
        
    
    elif FIG_NUM == 3:
        public['G_ub'] = np.append(public['G_ub'], [[-1,0]], axis = 0)
        public['h_ub'] = np.append(public['h_ub'], [-4], axis = 0)
        aux_line = extra2
        extra1 = {'G_ub': [[0, -1]],
            'h_ub': [-3],
            'x_lp': None,
            'z_lp': -12,}
        extra2 = {'G_ub': [[0, 1]],
            'h_ub': [2],
            'x_lp': [4,  2],
            'z_lp': -4,}
        
    elif FIG_NUM == 4:
        public['G_ub'] = np.append(public['G_ub'], [[1,0],[0,1]], axis = 0)
        public['h_ub'] = np.append(public['h_ub'], [3,2], axis = 0)
        aux_line = {'G_ub': [[-1,0], [0, -1]],
            'h_ub': [-4, -3],
            'x_lp': None,
            'z_lp': -12,}
        extra1 = {'G_ub': [[-1,0]],
            'h_ub': [-2],
            'x_lp': [2,  2],
            'z_lp': -6,}
        extra2 = {'G_ub': [[1,0]],
            'h_ub': [1],
            'x_lp': [1,  1.6],
            'z_lp': -5.4,}

    # 设置绘图范围
    x_range = np.linspace(0, 5.5, 400)
    y_range = np.linspace(0, 4, 400)
    # if FIG_NUM in [2]:
    #     x_range = np.linspace(0.3, 5.2, 400)
    #     y_range = np.linspace(1.6, 5, 400)
    # if FIG_NUM in [3]:
    #     x_range = np.linspace(0, 6, 400)
    #     y_range = np.linspace(-0.3, 1.5, 400)
    # if FIG_NUM in [4]:
    #     x_range = np.linspace(1.5, 5.2, 400)
    #     y_range = np.linspace(1.8, 5, 400)
    # if FIG_NUM in [5]:
    #     x_range = np.linspace(0.5, 1.2, 400)
    #     y_range = np.linspace(-0.1, 1.2, 400)
    # if FIG_NUM in [6]:
    #     x_range = np.linspace(-0.3, 1.2, 400)
    #     y_range = np.linspace(-0.1, 0.7, 400)
    # B3
    # x_range = np.linspace(-1, 5, 400)
    # y_range = np.linspace(-1, 4, 400)
    # B5
    # x_range = np.linspace(-5, 3, 400)
    # y_range = np.linspace(-9, 2, 400)
    X, Y = np.meshgrid(x_range, y_range)
    fig = plt.figure(figsize=(7, 7))
    # if FIG_NUM not in [3, 5]:
    #     fig = plt.figure(figsize=(7, 7))
    # elif FIG_NUM == 3:
    #     fig = plt.figure(figsize=(7, 3))
    # elif FIG_NUM == 5:
    #     fig = plt.figure(figsize=(4.5, 7))
    
    def draw_lp2D(public_pblm:dict, ex1:dict, ex2:dict, aux_line:dict=None):
        # 绘制线性规划问题的解和约束条件，但不绘制等高线
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        if FIG_NUM in [3,5,6]:
            plt.rcParams['font.size'] = 30  # 调整字号大小
        else:
            plt.rcParams['font.size'] = 40  # 调整字号大小

        colors = ['#FF7F0E','#1F77B4' , '#1F77B4', '#FF7F0E', '#1F77B4', '#FF7F0E', '#FF7F0E','#1F77B4' ,'#2CA02C', '#D62728', '#D62728', '#D62728', '#9467BD',
                  '#9467BD', '#D62728', '#9467BD',
                  '#D62728', '#9467BD', '#D62728', '#9467BD', '#D62728', '#9467BD',
                  '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BE7F', '#D62728', '#9467BD']
        colors.insert(0,'black')
        colors.insert(0,'black')
        colors.insert(0,'black')
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
            # plt.text(pos_x, pos_y, label, fontsize=60, color=font_color, fontweight='bold',style='italic')

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
            feasible_region(ex1['G_ub'],ex1['h_ub'], "2+", 3.2, 2.8,"Greys", "#ff8c00")
            feasible_region(G_ub2,h_ub2, "", 0.5, 3.0,"Blues", (31/255, 119/255, 180/255))
        elif FIG_NUM == 3:
            feasible_region(ex1['G_ub'],ex1['h_ub'], "  3+", 2.6, 0.2,"Greys", "#ff8c00")
            feasible_region(G_ub2,h_ub2, "3-", 0.2, 0.7,"Blues", (31/255, 119/255, 180/255))
            
            # plt.text(2.6, 0.2, "Integer", fontsize=40, color='r', fontweight='bold',style='italic')
            # plt.annotate(" ",
            #         xy=(0.9,0.1),
            #         xytext=(-20, 40),
            #         textcoords="offset points",
            #         arrowprops=dict(arrowstyle="->", color='r', linewidth=4),
            #         color='r',
            #         fontsize = 16,)
        elif FIG_NUM == 4:
            feasible_region(G_ub1,h_ub1, "4+", 3.4, 3.3,"Oranges", "#ff8c00")
            feasible_region(G_ub2,h_ub2, "4-", 2.8, 2.2,"Blues", (31/255, 119/255, 180/255))
            # plt.text(3.2, 3.3, "Integer", fontsize=60, color='r', fontweight='bold',style='italic')
            ax = fig.gca()
            # ax.add_patch(Ellipse((3.5, 2), width=3, height=0.2, 
            #             edgecolor=(31/255,119/255,180/255), facecolor='none', linestyle='--', linewidth=4))

        def draw_boundary(G_ub, h_ub, isPublic:bool=False):
            nonlocal cid
            
            lw = 1 if isPublic else 2
            # if isPublic:
            #     G_ub = G_ub[2:]
            #     h_ub = h_ub[2:]
            for i, ((x, y), h) in enumerate(zip(G_ub, h_ub)):
                print("cid: ", cid-3)
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
            if aux_line is not None:
                draw_boundary(aux_line['G_ub'], aux_line['h_ub'],True)
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
            mark_opt(public['x_lp'], public['z_lp'], -80,55, '')
        elif FIG_NUM == 1:
            mark_opt(ex1['x_lp'], ex1['z_lp'], -100,35, '-')
            mark_opt(ex2['x_lp'], ex2['z_lp'], 80,15, "+")
        elif FIG_NUM == 2:
            mark_opt(ex1['x_lp'], ex1['z_lp'], -120,25, '-')
            mark_opt(ex2['x_lp'], ex2['z_lp'],-170,-55, "+")
        elif FIG_NUM == 3:
            mark_opt(ex1['x_lp'], ex1['z_lp'], 20,55, '-')
            mark_opt(ex2['x_lp'], ex2['z_lp'],170,-100, "+")
        elif FIG_NUM == 4:
            mark_opt(ex1['x_lp'], ex1['z_lp'], 20,-55, '-', True)
            mark_opt(ex2['x_lp'], ex2['z_lp'],-120,20, "+")

        # 图表设置
        # plt.title('Linear Programming')
        # plt.xlabel('x')
        # plt.ylabel('y')
        plt.grid(True,linestyle='--', alpha=0.3)
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlim(x_range.min(), x_range.max())
        plt.ylim(y_range.min(), y_range.max())
        plt.xticks([0, 1, 2, 3, 4, 5])
        plt.yticks([0, 1, 2, 3, 4])
        # plt.legend(loc = "upper left",prop={'family': 'Times New Roman'})
        plt.savefig(f"E:\Files\A-blockchain\\branchbound\\figs\\fig1\\B{FIG_NUM}_{time.strftime('%m%d_%H%M%S')}.svg")
        plt.show()
        plt.close()
    draw_lp2D(public, extra1, extra2, aux_line)


def plot_bounds_fig1(file_path):
    # 读取并解析文件
    with open(file_path, 'r') as f:
        jsondata_list = f.read().split('\n')
        data_list = [dict(json.loads(js)) for js in jsondata_list]
    # 开始绘图
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 20  # 调整字号大小
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 20  # 调整字号大小

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
    columns = ["miner", "block", "round", "bround", "pname", "pre_pname", "ub", "fathomed", "allInteger", "isFork"]
    ub_data = [dict(zip(columns, point)) for point in ub_data]
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
    count_children(ub_data)

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
            s = 200
            print(pname)
        else:
            color = '#00B0F0'
            s = (children_counts[pname]**1.5)*100+1
        if ub_data[i].get("fathomed", False) and not is_integer:
            color = 'grey' 
            s = 150
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
                 'lightgrey', linestyle='--', linewidth = 1.5, zorder = 1)
    fig = plt.gcf()
    ax = fig.gca()
    # for mb, points in points_by_mb.items():
    #     patch = smooth_convex_hull(np.array(points))
    #     if patch:
    #         plt.gca().add_patch(patch)
    ub_df = pd.DataFrame(ub_data)
    ub_df['children_count'] = ub_df['pname'].apply(lambda x: children_counts[get_pname(x)])
    ub_df['pname'] = ub_df['pname'].apply(lambda x: get_pname(x))
    ub_df['pre_pname'] = ub_df['pre_pname'].apply(lambda x: get_pname(x) if x != "None" else None)
    ub_df['is_main'] = ~ub_df['isFork'] & ~ub_df['fathomed'] & (ub_df['block'] != "None")
    ub_df['is_fathomed'] = ub_df['fathomed']
    main_df = ub_df
    min_ub_df = main_df[main_df['ub'] == main_df.groupby('block')['ub'].transform('min')]
    max_ub_value = main_df.groupby('block')['ub'].max()
    block_df = pd.merge(min_ub_df, max_ub_value, on='block', suffixes=('_min', '_max'))
    blocks = []
    i=0
    for _, row in block_df.iterrows():
        if row['bround'] == -1:
            continue
        if row['block'] in blocks:
            continue
        i+=1
        # print(i)
        # print(row)
    # 提取每个block的数据
        blocks.append(row['block'])
        min_ub = row['ub_min']
        max_ub = row['ub_max']
        # 绘制圆角矩形
        width = 2 if i == 1 else 1.1
        # children_count = children_counts[row['pre_pname']]
        color = "#00B0F0"
        # color = my_blues(0 + 0.5*(1-point_norm(children_count)))  # 将颜色范围限制在0.3-0.7之间，使颜色更浅
        # left = row['bround']-width/2
        # 计算对数尺度下的矩形边界
        l = 1.2
        rect = Rectangle((row['bround']-width/2, min_ub-l), width, max_ub - min_ub+l*2, 
                        linewidth=1.5, edgecolor='#5494CE',  facecolor = color,
                        linestyle='-', capstyle='round', joinstyle='round',
                        rotation_point='center',alpha = 0.1, zorder = 0)
        ax.add_patch(rect)
    
    # 绘制Lowerbounds的线段
    plt.plot(lb_rounds_new, lb_values_new, color='black', linewidth=4, label='lowerbound', zorder = 0)

    plts = [plt.Line2D([], [], color='black', linewidth=2, label='lowerbound'),
            plt.Line2D([],[],color="red", linestyle='None',marker = 'o', label="integer"),
        plt.Line2D([],[],color="#00B0F0", linestyle='None',marker = 'o',  label="relaxed LP sol"),
        plt.Line2D([],[],color="grey", linestyle='None',marker = 'o',  label="fathomed"),]
        # plt.Line2D([100],[100],color="black", linestyle='None',marker = 'o', label="fork"),
        # plt.Line2D([100],[100],color="#9acd32", linestyle='None',marker = 'o', label="unpublished")]
    
    plt.xlabel(' ')
    plt.ylabel(' ')
    # plt.title('Scatter Plot with UB and Lowerbounds')
    plt.ylim(-10, -3) 
    plt.xlim(-1.3, 20) 
    # plt.xticks([0, 5, 10, 15])
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(handles = plts, loc = "lower left", bbox_to_anchor=(-0.31, -0.05))
    fig = plt.gcf()
    fig.subplots_adjust(left=0.244, right=0.995, top=0.971, bottom=0.016)
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#dddddd')
    plt.savefig(f"E:\Files\A-blockchain\\branchbound\\figs\\fig1\\bounds_{time.strftime('%m%d_%H%M%S')}.svg")
    plt.show()

if __name__ == "__main__":
    # for i in range(5):
    #     plot_feasible_regions_fig1(i)
    plot_bounds_fig1(Path.cwd() / "Results" / "20250326" / "161237" / "pfig1_03262m1d1evaluation results.json")

