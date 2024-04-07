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
    FancyBboxPatch,
    Patch,
    PathPatch,
    Rectangle,
)
from matplotlib.path import Path
from matplotlib.ticker import PercentFormatter
from scipy.signal import find_peaks, peak_prominences
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde

import background
import consensus.branchbound as bb

SAVE_PREFIX = "E:\Files\A-blockchain\\branchbound\\branchbound仿真\\0129"
pathlib.Path.mkdir(pathlib.Path(SAVE_PREFIX), exist_ok=True)
SAVE = True
def plot_relaxed_sulotions(data_list:list[dict]):
    def get_max_min(ubs:list[dict]):
        mn = 0
        mx = -sys.maxsize
        for ub in ubs:
            data = []
            for values in ub.values():
                data.extend(values)
            mn = min(data) if min(data) > mn else mn
            mx = max(data) if max(data) > mn else mn
        return mx, mn
    lower_bounds:list[dict] = []
    upper_bounds:list[dict] = []
    for data in data_list:
        upper_bounds.append(data["relax_sols"])
        lower_bounds.append(data["lowerbounds"])
    colors = ['#0072BD','#ff7f0e', '#77AC30','#D95319',]
    # colors = ['#5A84FF','#FFC642', '#6AC97E','#D95319',]
    markers=['o','o','o']
    lb_colors = ['#02447B','#ff4500','green','orange']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小
    max_rd = 0
    fig = plt.figure(figsize=(10, 6.5))
    max_ub,min_ub = get_max_min(upper_bounds)
    ax = brokenaxes(ylims=((0, 50), (500, max_ub)),despine=False, hspace=0.05,d=0.01)
    miner_nums = []
    for i, (lb,ub) in enumerate(zip(lower_bounds, upper_bounds)):
        miner_num = len(list(ub.values())[0])
        miner_nums.append(miner_num)
        label = [f'{miner_num}miner sulotions', f'{miner_num}miner lowerbound']
        ub_rounds = [int(r) for r in ub.keys()]
        lb_rounds = [int(r) for r in lb.keys()]
        max_rd = max(max_rd, max(ub_rounds))
        # ax = fig.add_subplot(1,3, plt_index+1)
        scatter_datas = defaultdict(list)
        for r, values in ub.items():
            for m, value in enumerate(values):
                scatter_datas[m].append(value)
        for scatter_data in scatter_datas.values():
            ax.scatter(ub_rounds, scatter_data, c=colors[i],alpha=0.2, s =3,zorder = i)
        # zorder=2 if miner_num == 1 else 2
        # ax.scatter(ub_rounds, scatter_datas[0], c=colors[i],alpha=0.2, s =8,zorder = i,marker =markers[i])

        ax.plot(lb_rounds, [v if v != 0 else v+4 for v in lb.values()], c=lb_colors[i], linewidth = 4, alpha=0.9,zorder = i)
    
    # ax = fig.gca()
    ax.set_xlim(0, max_rd+20)
    ax.set_xlabel('Round',20)
    ax.set_ylabel('Relaxed Solutions',40)
    x_tick_step = 1000
    x_ticks = np.arange(0, max_rd, x_tick_step)
    # x_ticks = np.insert(x_ticks, 0, min(data))
    ax.set_xticks(x_ticks)
    lgdhdls = []
    lgdhdls.append(plt.Line2D([100],[100],color="grey", alpha=0.4, linestyle='None',marker = 'o', markersize=7, label="relaxed solutions"))
    lgdhdls.append(plt.Line2D([],[],color="grey",label="lower-bounds",linewidth = 4))
    lgdhdls.extend([Patch(facecolor=colors[i], label=f'miner_num = {miner_num}') for i, miner_num in enumerate(miner_nums)])
    # ax.legend(handles, labels)
    plt.legend(handles=lgdhdls, loc='lower right')
    ax.grid()
    ax.set_rasterized(True)
    if SAVE:
        plt.savefig(SAVE_PREFIX + "\收敛v100m135 3.eps", dpi=300)
    plt.show()
    
def plot_bounds_fig3(data_list:list[dict]):
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

    def draw_int_path(path):
        for i in range(len(path) - 1):
            start_point = (path[i]['bround'], path[i]['ub'])
            end_point = (path[i + 1]['bround'], path[i + 1]['ub'])
            ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                    color = "#FF8283", linestyle='-', linewidth=2, alpha=0.7, zorder=6)


    def get_pre_root(point):
        if point['pre_pname'] == None:
            return None
        pre_point_rows = ub_df[ub_df['pname'] == point['pre_pname']]
        pre_point = pre_point_rows.iloc[0] if not pre_point_rows.empty else None
        while pre_point is not None and pre_point['block'] == point['block']:
            point = pre_point
            pre_point_rows = ub_df[ub_df['pname'] == point['pre_pname']]
            pre_point = pre_point_rows.iloc[0] if not pre_point_rows.empty else None
        if pre_point is None:
            return None 
        return root_df[root_df['block'] == pre_point['block']].iloc[0]

    
    data = data_list[0]
    ub_data = data.get("ubdata", [])
    columns = ["miner", "block", "round", "bround", "pname", "pre_pname", "ub", "fathomed", "allInteger", "isFork"]
    ub_data = [dict(zip(columns, point)) for point in ub_data]
    # for point in ub_data:
    #     if point["ub"] is None:
    #         continue
    #     point["ub"] = -point["ub"]
    lowerbounds = data.get("lowerbounds", {})
    point_lookup = {get_pname(point['pname']): point for point in ub_data}
    children_counts = defaultdict(lambda : 1) 
    count_children(ub_data)
    print(children_counts)

    # 创建数据帧
    ub_df = pd.DataFrame(ub_data)
    # ub_df['ub'] = -ub_df['ub']
    ub_df['children_count'] = ub_df['pname'].apply(lambda x: children_counts[get_pname(x)])
    ub_df['pname'] = ub_df['pname'].apply(lambda x: get_pname(x))
    ub_df['pre_pname'] = ub_df['pre_pname'].apply(lambda x: get_pname(x) if x != "None" else None)
    ub_df['is_main'] = ~ub_df['isFork'] & ~ub_df['fathomed'] & (ub_df['block'] != "None")
    ub_df['is_fathomed'] = ub_df['fathomed']
    # sampled_df = ub_df.sample(frac=0.01, random_state=0)  # 调整抽样比例
    root_df = ub_df.copy().groupby('block').apply(lambda x: x.nsmallest(1, 'round')).reset_index(1,drop=True)
    main_df = ub_df[ub_df["fathomed"] == False].copy()
    min_ub_df = main_df[main_df['ub'] == main_df.groupby('block')['ub'].transform('min')]
    max_ub_value = main_df.groupby('block')['ub'].max()
    block_df = pd.merge(min_ub_df, max_ub_value, on='block', suffixes=('_min', '_max'))
    # block_df = main_df.groupby('block').agg(min_ub=('ub', 'min'), max_ub=('ub', 'max'), bround=('bround', 'min')).reset_index()
    pre_rows = []
    for _, row in root_df.iterrows():
        pre_pname = row['pre_pname']
        matched_row = ub_df[ub_df['pname'] == pre_pname]
        pre_rows.append(matched_row)
    pre_df = pd.concat(pre_rows)
    bpre_rows = []
    for _, row in block_df.iterrows():
        pre_pname = row['pre_pname']
        matched_row = ub_df[ub_df['pname'] == pre_pname]
        bpre_rows.append(matched_row)
    bpre_df = pd.concat( bpre_rows)
    print(ub_df)
    print(block_df)
    print(root_df)
    print(pre_df)

    # 标记allInteger路径
    int_paths = []
    intpath_points = set()
    for point in ub_data:
        if point['allInteger'] and point["block"]!= "None":
            cur_point = point
            path = [cur_point]
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

    # sns.set(style="white")
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()

    for path in int_paths:
        draw_int_path(path)
    sampled_points = set(root_df['pname']).union(intpath_points).union((((0,0),0)))
    # sampled_points = set(ub_df.sample(frac=0.1, random_state=0)['pname'])
    # sampled_points = sampled_points.union(intpath_points)
    # ["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"] ["#FF8283", "#0D898A","#f9cc52","#5494CE", ] '#00796B' '#ff8c00' '#b22222'
    # 绘制UB数据点和连接线
    smain = 50
    s = 5
    sopt = 100
    rasterized=False
    sns.scatterplot(x="bround",y ="ub",data = ub_df[(ub_df["fathomed"] == False) & (ub_df["block"]!= "None")] ,
                    s = smain, color = '#0072BD', rasterized=rasterized ,edgecolor="none",zorder = 5, alpha = 0.7) ,
    sns.scatterplot(x="round",y ="ub",data = ub_df[(ub_df["fathomed"]== True) & (ub_df["allInteger"]==False) & (ub_df["block"]!= "None")] , 
                    color = "#ff8c00", s= s,rasterized=rasterized,edgecolor="none",alpha = 0.5)
    sns.scatterplot(x="round",y ="ub",data = ub_df[(ub_df["block"]== "None")], 
                    color = "#9acd32", s= 30,rasterized=rasterized,edgecolor="none",zorder = 4, alpha = 0.5)
    sns.scatterplot(x="bround",y ="ub",data = ub_df[(ub_df["isFork"]== True) & (ub_df["block"]!= "None")] , 
                    color = "black", s= 30,rasterized=rasterized,edgecolor="none",zorder = 4, alpha = 0.5)
    sns.scatterplot(x="bround",y ="ub",data = ub_df[(ub_df["allInteger"] == True) & (ub_df["block"]!= "None")] , 
                    color = "r", s = sopt,rasterized=rasterized,edgecolor="none",zorder = 6) 
    point_norm = mcolors.Normalize(vmin=0, vmax=bpre_df['children_count'].max())
    print(bpre_df['children_count'].max())
    blues = plt.cm.Blues
    # my_blues = mcolors.LinearSegmentedColormap.from_list("my_blues", 
    #["#caf0f8","#caf0f8","#ade8f4","#90e0ef","#48cae4","#00b4d8","#0096c7", "#0077b6","#003049"])#"#caf0f8" ,
    drawRect = True
    if drawRect:
        blocks = []
        i=0
        for _, row in block_df.iterrows():
            if row['bround'] == -1:
                continue
            if row['block'] in blocks:
                continue
            i+=1
            print(i)
        # 提取每个block的数据
            blocks.append(row['block'])
            min_ub = row['ub_min']
            max_ub = row['ub_max']
            # 绘制圆角矩形
            width = 30
            children_count = children_counts[row['pre_pname']]
            color = blues(1- point_norm(children_count))
            rect = Rectangle((row['bround']-width/2, min_ub), width, max_ub - min_ub, 
                            linewidth=1.5, edgecolor='#5494CE',  facecolor = color,#facecolor='#00b4d8',
                            linestyle='-', capstyle='round', joinstyle='round',
                            rotation_point='center',alpha = 0.4, zorder = 2)
            # rect = FancyBboxPatch((row['bround'] - 500, min_ub), 
            #                   1000, max_ub - min_ub, 
            #                   boxstyle="round,pad=0.2,rounding_size=100", linewidth=1, 
            #                   edgecolor='none', facecolor='#0096c7', linestyle='--')
            plt.gca().add_patch(rect)
        for idx, block_row in block_df.iterrows():
            # 获取block点的坐标
            block_round = block_row['bround']
            block_ub = block_row['ub_min']

            # 获取pre点的坐标
            pre_row = pre_df[pre_df['pname'] == block_row['pre_pname']]
            if not pre_row.empty:
                pre_round = pre_row.iloc[0]['bround']
                pre_ub = pre_row.iloc[0]['ub']
                # 绘制折线连接两点
                plt.plot([pre_round, block_round, block_round], [pre_ub, pre_ub, block_ub], color = '#5494CE',
                        linestyle='--',linewidth = 1.5,alpha = 0.5, zorder = 0)#color='#CFCFCF'

    # # 处理Lowerbounds数据
    lb_rounds_new = list(map(int, lowerbounds.keys()))
    lb_values_new = [v for v in lowerbounds.values()]
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
    
    # for i, point in ub_df.iterrows():
    #     if point['pname'] not in sampled_points:
    #         continue  # 只绘制抽样的点
    # #     # if point['block'] != "B1":
    # #     #     continue
    # #     # print(point['pname'])
    # #     # if not point["allInteger"]:
    # #     #     continue
    # #     # if point['pname'] in sampled_points or (point["fathomed"] and not point["allInteger"]):
    #     pre_point = get_pre_point(point) if not point["allInteger"] else point
    #     draw_point(point, pre_point)

    # # 绘制Lowerbounds的线段
    plt.plot(lb_rounds_new, lb_values_new, color="#66A266", linewidth=8, zorder = 3)

    plts = [plt.Line2D([], [], color='green', linewidth=2, label='upper bounds'),
        plt.Line2D([],[],color="red", linestyle='None',marker = 'o', label="integer"),
        plt.Line2D([],[],color="#0072BD", linestyle='None',marker = 'o',  label="main-chain"),
        plt.Line2D([],[],color="orange", linestyle='None',marker = 'o',  label="fathomed"),
        plt.Line2D([],[],color="black", linestyle='None',marker = 'o', label="fork"),
        plt.Line2D([],[],color="#9acd32", linestyle='None',marker = 'o', label="unpublished")]
    
    plt.xlabel('Round')
    plt.ylabel('Values')
    plt.xscale("log")
    plt.xlim(100, 2500)
    # plt.ylim(2500, 4000)
    fig.subplots_adjust(left=0.079, bottom=0.1, right=0.98, top=0.98)
    plt.ylim(59, 63)
    # plt.ylim(180, 250)
    # plt.xlim(-30, 430)
    # plt.title('Scatter Plot with UB and Lowerbounds')
    # plt.grid(True)
    # plt.legend(handles = plts)
    # ax.set_rasterized(True)
    plt.savefig(SAVE_PREFIX + f"\\bounds_maxsat{time.strftime('%H%M%S')}.svg", dpi=300)
    # plt.show()

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

def plot_bounds2(data_list:list[dict]):
    # 读取并解析文件
    # with open(file_path_new, 'r') as file:
    #     json_data_new = json.load(file)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小
    
    data = data_list[0]
    ub_data = data.get("ubdata", [])
    lowerbounds = data.get("lowerbounds", {})

    rounds = []
    brounds = []
    ubs = []
    pnames = []
    pre_pnames = []
    children_counts = defaultdict(lambda : 1) 
    def smooth_convex_hull(points):
        """
        Generate a smooth shape around the given points.
        """
        if len(points) < 3:
            # Not enough points for a convex hull
            return None

        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Creating a Path object to encapsulate the convex hull points
        codes = [Path.MOVETO] + [Path.CURVE4] * (len(hull_points) - 1)
        path = Path(hull_points, codes)

        # Creating a patch from the path
        patch = PathPatch(path, facecolor='orange', lw=1, alpha=0.3, edgecolor='none')
        return patch

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
                pre_point = next((p for p in ub_data if get_pname(p["pname"]) == pre_pname), None)
                if pre_point is None:
                    break
                pre_pname = get_pname(pre_point["pre_pname"])
                
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

    lb_rounds_new = list(map(int, lowerbounds.keys()))
    lb_values_new = list(lowerbounds.values())

    plt.figure(figsize=(12, 3))

    # 绘制UB数据点和连接线
    for i, pname in enumerate(pnames):
        # if rounds[i]==0:
        #     continue
        
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
        if ub_data[i].get("isFork", False):
            color = "black"
            alpha = 1
            s = 10
        if ub_data[i].get("block", "None") == "None":
            color = "#9acd32"
            alpha = 0.5
            round = rounds[i]
            s = 10
        else:
            round = brounds[i]
            # alpha = 0.5
        plt.scatter(round, ubs[i], color=color, alpha=alpha, s=s ,zorder = 3)
        # ax.scatter(ub_rounds, scatter_data, c=colors[i],alpha=0.2, s =3)

        # 连接前一个点，使用浅色线条
        if pre_pnames[i]:
            pre_index = pnames.index(pre_pnames[i])
            if ubs[pre_index] is not None:  # 确保前一个点的ub值非空
                # plt.plot([rounds[i], rounds[pre_index]], 
                #             [ubs[i], ubs[pre_index]], 'lightgrey', linestyle='--',
                #             linewidth = 0.5, zorder = 0)
                plt.plot([brounds[i], brounds[pre_index]], 
                          [ubs[i], ubs[pre_index]], 'lightgrey', linestyle='--',
                            linewidth = 1, zorder = 0)

    # 绘制Lowerbounds的线段
    plt.plot(lb_rounds_new, lb_values_new, color='green', linewidth=2, label='Lowerbounds', zorder = 0)

    plts = [plt.Line2D([], [], color='green', linewidth=2, label='Lowerbounds'),
        plt.Line2D([100],[100],color="red", linestyle='None',marker = 'o', label="integer solution"),
        plt.Line2D([100],[100],color="#0072BD", linestyle='None',marker = 'o',  label="main chain"),
        plt.Line2D([100],[100],color="orange", linestyle='None',marker = 'o',  label="fathomed"),]
        # plt.Line2D([100],[100],color="black", linestyle='None',marker = 'o', label="fork"),
        # plt.Line2D([100],[100],color="#9acd32", linestyle='None',marker = 'o', label="unpublished")]
    
    plt.xlabel('Round')
    plt.ylabel('Values')
    # plt.title('Scatter Plot with UB and Lowerbounds')
    plt.ylim(-25, -5) 
    plt.xlim(-1, 25) 
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

def plot_average_convergence(data_list, ylabel, label_list, ylim):
    max_data = 0
    min_data = 0
    colors = ['#0072BD','#ffa500','#62bd00','#bd0082', '#bd9a00','#D95319', '#00b1bd', 
            '#8a2be2', '#00008b', '#000000','#ff8c00','#f08080','#cc6600']
    fig = plt.figure(figsize=(10, 6.5))
    plts = []
    for plt_index, data in enumerate(data_list):
        l1,  = plt.plot(
            list(range(1,len(data)+1)), 
            data, 
            colors[plt_index],
            label = label_list[plt_index])
        plts.append(l1)
        plt.plot(
            list(range(1,len(data)+1)), 
            [data[len(data)-1] for _ in range(len(data))], 
            colors[plt_index], 
            linestyle = '-',
            alpha=0.2)
        max_data = max(data) if max(data)>max_data else max_data
        min_data = min(data) if min(data)<min_data else min_data
    # plt.ylim(min_data-20, max_data+20)
    plt.ylim(ylim[0], ylim[1])
    ax = fig.gca()
    for label in ax.xaxis.get_ticklabels():
        label.set_rotation(30)
    plt.xlabel('Round')
    plt.ylabel(ylabel)
        #added this to get the legend to work
    # handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles = plts, loc='lower right')
    plt.show()
    result_path =  background.get_result_path()
    plt.savefig(result_path / f'{ylabel}_{label_list[0]}.svg')
    plt.close()

def plot_line_chart_miner_as_xlabel2(data_dict, ylabel, ylim, entry):
    """
    data_dict格式实例: 
        {(dmin, miner_num): 
        {10: {'ave_mb_forkrate': 0.4166666666666667,
            'ave_solve_round': 33.6,
            'ave_subpair_num': 13.11111111111111,
            'total_mb_forkrate': 0.46},
        15: {'ave_mb_forkrate': 0.40626984126984117,
            'ave_solve_round': 29.7,
            'ave_subpair_num': 12.0,
            'total_mb_forkrate': 0.42},
        20: {'ave_mb_forkrate': 0.311038961038961,
            'ave_solve_round': 27.3,
            'ave_subpair_num': 14.222222222222221,
            'total_mb_forkrate': 0.47368421052631576}}
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小
    x_range = []
    ydata_dict = {} # [[对单个difficulty的数据]]
    ydata_miner = []
    legend_list = []

    for (var_num, difficulty), miner_data in data_dict.items():
        if difficulty not in [3,5,8,10]:
            continue
        if var_num not in ydata_dict.keys():
            ydata_dict.update({var_num:[]})
        legend_list.append(f"difficulty = {int(difficulty)}")
        print(legend_list)
        x_range = []
        ydata_miner = []
        for miner_num, data in miner_data.items():
            x_range.append(int(miner_num))
            ydata_miner.append(data[entry])
        ydata_dict[var_num].append(ydata_miner)

    max_data = 0
    min_data = 200
    # colors = ['#0072BD','#ffa500','#77AC30']
    # colors = ['r','b','g']
    # colors = ['#0072BD','#ffa500','#77AC30', '#62bd00','#bd0082', '#bd9a00','#00b1bd',
    #         '#D95319', '#cc6600','#8a2be2', '#00008b', '#000000','#ff8c00','#f08080']
    fig = plt.figure(figsize=(10, 6.5))
    plts = []
    for i, (var_num, data_list) in enumerate(ydata_dict.items()):
        if var_num == 20:
            linestyle = '-'
        else:
            linestyle = '--'
        for plt_index, data in enumerate(data_list):
            l1,  = plt.plot(
                x_range, 
                data,
                label=legend_list.pop(0), 
                linestyle = linestyle, 
                marker = 'o')
                # color = colors[plt_index])
            plts.append(l1)
            max_data = max(data) if max(data)>max_data else max_data
            min_data = min(data) if min(data)<min_data else min_data
            print(min_data, min(data))
    # plt.ylim(min_data-20, max_data+20)
    plt.ylim(ylim[0],ylim[1])
    ax = fig.gca()
    plt.xlabel('Number of miners')
    plt.ylabel(ylabel)
    #added this to get the legend to work
    # handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles = plts, loc='best')
    x_ticks = np.arange(0, max(x_range) + 1, 1)
    # x_ticks = np.insert(x_ticks, 0,0)
    ax.set_xticks(x_ticks)
    ax.grid()
    plt.show()
    # result_path =  global_var.get_result_path()
    # plt.savefig(result_path / f'{ylabel}_{legend_list[0]}.svg')
    plt.close()

def plot_line_chart_miner_workload(data_dict, ylabel, ylim, withunpub):
    """
    data_dict格式实例: 
        {(var_num, difficulty): 
        {miner_num: {'subpair_rate': 0.4166666666666667,
            'subunpub_rate': 33.6},
        15: {'subpair_rate': 0.40626984126984117,
            'subunpub_rate': 29.7},
        20: {'subpair_rate': 0.311038961038961,
            'subunpub_rate': 27.3}}
    """
    def default_list():
        return []
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小W
    x_range = []
    data_groups = defaultdict(default_list)
    unpub_data_groups = defaultdict(default_list)
    acp_data_groups = defaultdict(default_list)
    legend_list = []
    # 读取数据
    for (var_num, difficulty), data_group in data_dict.items():
        # if difficulty not in [3,5,8,10]:
        if difficulty not in ['BFS','DFS','BrFS','Rand']:
            continue
        legend_list.append(f"strategy = {difficulty}")
        # legend_list.append(f"difficulty = {int(difficulty)}")
        x_range = []
        data_lines = []
        unpub_data_lines = []
        acp_data_lines = []
        for miner_num, data_point in data_group.items():
            x_range.append(int(miner_num))
            # data_lines.append(data_point['subpair_rate'])
            # unpub_data_lines.append(data_point['subunpub_rate'])
            data_lines.append(data_point['ave_subpair_num']/miner_num)
            
            acp_data_lines.append(data_point['ave_acp_subpair_num']/miner_num)
            unpub_data_lines.append((data_point['ave_subpair_unpubs']-data_point['ave_acp_subpair_num'])/miner_num)
        data_groups[var_num].append(data_lines)
        unpub_data_groups[var_num].append(unpub_data_lines)
        acp_data_groups[var_num].append(acp_data_lines)

    max_data = 0
    min_data = 200
    colors = ['#0072BD','#ffa500','#32cd32','#b22222','#9467bd', '#87cefa', '#ff6347']
    makers = ['o','^','x','v']
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.gca()
    plts = []
    for i, (var_num, data_list) in enumerate(acp_data_groups.items()):
        for plt_index, data in enumerate(data_list):
            l1,  = ax.plot(
                x_range, 
                data,
                label=legend_list[plt_index], 
                linestyle = '-', 
                marker = makers[plt_index],
                color = colors[plt_index])
            plts.append(l1)
            max_data = max(data) if max(data)>max_data else max_data
            min_data = min(data) if min(data)<min_data else min_data
            print(min_data, min(data))
    if withunpub:
        ax2 = ax.twinx()
        for i, (var_num, data_list) in enumerate(unpub_data_groups.items()):
            for plt_index, data in enumerate(data_list):
                l1,  = ax2.plot(
                    x_range[1:], 
                    data[1:],
                    label=legend_list[plt_index],
                    linestyle = '--', 
                    marker = makers[plt_index],
                    color = colors[plt_index])
                # plts.append(l1)
                max_data = max(data) if max(data)>max_data else max_data
                min_data = min(data) if min(data)<min_data else min_data
                print(min_data, min(data))
        # for i, (var_num, data_list) in enumerate(acp_data_groups.items()):
        #     for plt_index, data in enumerate(data_list):
        #         l1,  = plt.plot(
        #             x_range, 
        #             data,
        #             label=legend_list[plt_index],
        #             linestyle = '-.', 
        #             marker = 'x',
        #             color = colors[plt_index])
        #         # plts.append(l1)
        #         max_data = max(data) if max(data)>max_data else max_data
        #         min_data = min(data) if min(data)<min_data else min_data
        #         print(min_data, min(data))
    plts.append(plt.Line2D([0], [0], linestyle = '-', color='grey', label="Effective"))
    plts.append(plt.Line2D([0], [0], linestyle = '--', color='grey', label="Wasted"))
    # plts.append(plt.Line2D([0], [0], marker='^', linestyle = '--', color='grey', label="workload with unpublished"))
    # 创建椭圆并添加到图中
    ellipse_subpair = Ellipse((15, 89), width=0.5, height=160, edgecolor='#00796B', facecolor='none', linestyle='--', linewidth=2)
    ellipse_subunpub = Ellipse((6, 800), width=0.5, height=200, edgecolor='#ff8c00', facecolor='none', linestyle='--', linewidth=2)

    # 添加文本框和箭头
    ax.add_patch(ellipse_subpair)
    ax.add_patch(ellipse_subunpub)
    ax.annotate("Effective workload per miner",
                xy=(15, 160),
                xytext=(-150, 30),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color='#00796B', linewidth=2),
                color='#00796B',
                fontsize = 16,
                fontweight='bold')

    ax.annotate("Wasted workload per miner",
                xy=(6, 800),
                xytext=(2, -50),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color='#ff8c00', linewidth=2),
                color='#ff8c00',
                fontsize = 16,
                fontweight='bold')

    ax.annotate(" ",
                xy=(7, 420),
                xytext=(55, 45),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color='#ff8c00', linewidth=2),
                color='#ff8c00',
                fontsize = 16,
                fontweight='bold')
    ax.spines['left'].set_color('#00796B')  # Set the color of the y-axis to blue
    ax.yaxis.label.set_color('#00796B')
    ax.tick_params(axis='y', colors='#00796B')
    ax2.yaxis.label.set_color('#ff8c00')
    ax2.tick_params(axis='y', colors='#ff8c00')
    ax2.spines['right'].set_color('#ff8c00')  # Set the color of the second y-axis to green

    ax.set_ylim(0,1350)
    ax2.set_ylim(0,500)
    ax.set_ylabel("Effective workload per miner")
    ax2.set_ylabel("Wasted workload per miner")
    ax.set_xlabel('Number of miners')
    
    x_ticks = np.arange(0, max(x_range) + 1, 1)
    # x_ticks = np.insert(x_ticks, 0,0)
    ax.set_xticks(x_ticks)
    ax.legend(handles = plts, loc='best')
    ax.grid()
    plt.show()
    plt.close()

def plot_line_chart_difficulty_as_xlabel(data_dict, ylabel, ylim, entry):
    """
    data_dict格式实例: 
        {(var_num, miner_num): 
        {10: {'subpair_rate': 0.4166666666666667,
            'subunpub_rate': 33.6},
        15: {'subpair_rate': 0.40626984126984117,
            'subunpub_rate': 29.7},
        20: {'subpair_rate': 0.311038961038961,
            'subunpub_rate': 27.3}}
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小
    x_range = []
    ydata_dict = {} # [[对单个difficulty的数据]]
    ydata_list_single_d = []
    legend_list = []

    for (var_num, miner_num), data_dict_single_m in data_dict.items():
        if var_num not in ydata_dict.keys():
            ydata_dict.update({var_num:[]})
        legend_list.append(f"miner_num = {int(miner_num)}")
        print(legend_list)
        x_range = []
        ydata_list_single_d = []
        for difficulty, subdict_data in data_dict_single_m.items():
            if difficulty in [3,5,8,10]:
                x_range.append(int(difficulty))
                ydata_list_single_d.append(subdict_data[entry])
        ydata_dict[var_num].append(ydata_list_single_d)

    max_data = 0
    min_data = 200
    # colors = ['#0072BD','#ffa500','#77AC30']
    # colors = ['r','b','g']
    colors = ['#0072BD','#ffa500','#32cd32','#b22222','#9467bd', '#87cefa', '#ff6347',
            '#bcbd22','#00b1bd','#77AC30', '#62bd00','#bd0082', '#bd9a00','#00b1bd',
            '#D95319', '#cc6600','#8a2be2', '#00008b', '#000000','#ff8c00','#f08080']
    fig = plt.figure(figsize=(10, 6.5))
    plts = []
    linestyles = ['-', '--', '-.']
    for i, (var_num, data_list) in enumerate(ydata_dict.items()):
        linestyle = linestyles[i]
        for plt_index, data in enumerate(data_list):
            l1,  = plt.plot(
                x_range, 
                data,
                label=legend_list.pop(0), 
                linestyle = linestyle, 
                marker = 'o',
                color = colors[plt_index])
                # color = colors[plt_index])
            plts.append(l1)
            max_data = max(data) if max(data)>max_data else max_data
            min_data = min(data) if min(data)<min_data else min_data
            print(min_data, min(data))
    # plt.ylim(min_data-20, max_data+20)
    plt.ylim(ylim[0],ylim[1])
    ax = fig.gca()
    plt.xlabel('Difficulty level')
    plt.ylabel(ylabel)
    #added this to get the legend to work
    # handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles = plts, loc='best')
    ax.grid()
    plt.show()
    # result_path =  global_var.get_result_path()
    # plt.savefig(result_path / f'{ylabel}_{legend_list[0]}.svg')
    plt.close()

def plot_line_chart_difficulty_subpair_subunpub(data_dict, ylabel, ylim, withunpub):
    """
    data_dict格式实例: 
        {(var_num, miner_num): 
        {difficulty: {'subpair_rate': 0.4166666666666667,
                      'subunpub_rate': 33.6},...}
    """
    def default_list():
        return []
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小
    x_range = []
    data_groups = defaultdict(default_list)
    unpub_data_groups = defaultdict(default_list)
    legend_list = []
    # 读取数据
    for (var_num, miner_num), data_group in data_dict.items():
        # legend_list.append(f"var_num = {int(diff_key[0])}, miner_num = {int(diff_key[1])}")
        # legend_list.append(f"var_num = {int(var_num)}, miner_num = {int(difficulty)}")
        legend_list.append(f"miner_num = {int(miner_num)}")
        x_range = []
        data_lines = []
        unpub_data_lines = []
        for difficulty, data_point in data_group.items():
            if difficulty not in [3,5,8,10]:
                continue
            x_range.append(int(difficulty))
            data_lines.append(data_point['subpair_rate'])
            unpub_data_lines.append(data_point['subunpub_rate'])
        data_groups[var_num].append(data_lines)
        unpub_data_groups[var_num].append(unpub_data_lines)

    max_data = 0
    min_data = 200
    colors = [
        '#0072BD','#ffa500','#32cd32','#b22222','#9467bd', '#87cefa', '#ff6347',
            '#bcbd22','#00b1bd','#0072BD','#ffa500','#77AC30', '#62bd00','#bd0082', '#bd9a00','#00b1bd',
            '#D95319', '#cc6600','#8a2be2', '#00008b', '#000000','#ff8c00','#f08080']
    fig = plt.figure(figsize=(10, 6.5))
    plts = []
    for i, (var_num, data_list) in enumerate(data_groups.items()):
        for plt_index, data in enumerate(data_list):
            l1,  = plt.plot(
                x_range, 
                data,
                label=legend_list[plt_index], 
                linestyle = '-', 
                marker = 'o',
                color = colors[plt_index])
            plts.append(l1)
            max_data = max(data) if max(data)>max_data else max_data
            min_data = min(data) if min(data)<min_data else min_data
            print(min_data, min(data))
    if withunpub:
        for i, (var_num, data_list) in enumerate(unpub_data_groups.items()):
            for plt_index, data in enumerate(data_list):
                l1,  = plt.plot(
                    x_range, 
                    data,
                    label=legend_list[plt_index],
                    linestyle = '--', 
                    marker = 'o',
                    color = colors[plt_index])
                # plts.append(l1)
                max_data = max(data) if max(data)>max_data else max_data
                min_data = min(data) if min(data)<min_data else min_data
                print(min_data, min(data))
    ax = fig.gca()
    # 创建椭圆并添加到图中
    ellipse_subpair = Ellipse((4.5, 1.2), width=0.2, height=0.45, edgecolor='#00796B', facecolor='none', linestyle='--', linewidth=2)
    ellipse_subunpub = Ellipse((6.5, 0.42), width=0.2, height=0.5, edgecolor='#ff8c00', facecolor='none', linestyle='--', linewidth=2)

    # 添加文本框和箭头
    ax.add_patch(ellipse_subpair)
    ax.add_patch(ellipse_subunpub)
    plt.annotate(
        "Workload in blockchain",
        xy=(4.5, 1.3),
        xytext=(10, 46),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color='#00796B', linewidth=2),
        color='#00796B',
        fontsize = 16,
        fontweight='bold'
    )

    plt.annotate(
        "Workload per miner",
        xy=(6.45, 0.6),
        xytext=(-30, 30),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color='#ff8c00', linewidth=2),
        color='#ff8c00',
        fontsize = 16,
        fontweight='bold'
    )
    plt.ylim(ylim[0],ylim[1])
    plt.xlabel('Difficulty level')
    plt.ylabel(ylabel)
    ax.legend(handles = plts, loc='best')
    ax.grid()
    plt.show()
    plt.close()

def plot_line_chart_miner_as_xlabel(data_dict, ylabel, ylim, entry):
    """
    data_dict格式实例: 
        {(var_num, difficulty): 
        {10: {'ave_mb_forkrate': 0.4166666666666667,
            'ave_solve_round': 33.6,
            'ave_subpair_num': 13.11111111111111,
            'total_mb_forkrate': 0.46},
        15: {'ave_mb_forkrate': 0.40626984126984117,
            'ave_solve_round': 29.7,
            'ave_subpair_num': 12.0,
            'total_mb_forkrate': 0.42},
        20: {'ave_mb_forkrate': 0.311038961038961,
            'ave_solve_round': 27.3,
            'ave_subpair_num': 14.222222222222221,
            'total_mb_forkrate': 0.47368421052631576}}
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小
    fig = plt.figure(figsize=(10, 6.5))
    ax1 = fig.gca()
    if entry == 'ave_solve_round':
        ax2 = ax1.twinx()
    """x_range = []
    ydata_list = [] # [[对单个difficulty的数据]]
    legend_list = []

    for diff_key, data_dict_single_d in data_dict.items():
        legend_list.append(f"var_num = {int(diff_key[0])}, difficulty = {int(diff_key[1])}")
        x_range = []
        ydata_list_single_d = []
        for miner_num, subdict_data in data_dict_single_d.items():
            x_range.append(int(miner_num))
            ydata_list_single_d.append(subdict_data[entry])
        ydata_list.append(ydata_list_single_d)

    max_data = 0
    min_data = 200
    colors = ['#0072BD','#ffa500','#32cd32','#b22222','#9467bd', '#87cefa', '#ff6347',
            '#bcbd22','#00b1bd',  '#77AC30', '#62bd00','#bd0082', '#bd9a00','#00b1bd',
            '#D95319', '#cc6600','#8a2be2', '#00008b', '#000000','#ff8c00','#f08080']
    fig = plt.figure(figsize=(10, 6.5))
    plts = []
    for plt_index, data in enumerate(ydata_list):
        l1,  = plt.plot(
            x_range, 
            data,
            label=legend_list[plt_index], 
            linestyle = '-', 
            marker = 'o',
            color = colors[plt_index])
        plts.append(l1)
        max_data = max(data) if max(data)>max_data else max_data
        min_data = min(data) if min(data)<min_data else min_data
        print(min_data, min(data))"""
    markers = ['o', '^', 'x', 'v', '*', 'D'] # 循环使用不同的标记
    colors = ['#0072BD','#ffa500','#32cd32','#b22222','#9467bd', '#87cefa', '#ff6347']
    linestyles = ['-', '--', '-.']
    reduce_colors = ['r','b']
    reduce_markers = ['o','x']
    # 对每一组var_num和difficulty进行迭代
    difficulties = [3,5,8,10]
    for i, (var_num, grouped_data) in enumerate(groupby(sorted(data_dict.keys(), key=lambda x: x[0]), key=lambda x: x[0])):
        # 对于每一种难度等级，绘制一条线
        for j, difficulty in enumerate(sorted([d for (v,d) in grouped_data if d in difficulties])):
            x = []
            y = []
            reduced_sol_rounds = []
            for (v, d), sub_data in data_dict.items():
                if not(v == var_num and d == difficulty):
                    continue
                for miner_num, metrics in sorted(sub_data.items()):
                    x.append(miner_num)
                    y.append(metrics[entry])
                    if  entry != 'ave_solve_round':
                        continue
                    if miner_num == 1:
                        reduced_sol_rounds.append(0)
                        continue
                    reduced_sol_rounds.append(1- metrics[entry]/sub_data[1][entry])
            ax1.plot(x, y, color = colors[j], marker=markers[j], linestyle = linestyles[i], markerfacecolor='none')
            if  entry == 'ave_solve_round' and difficulty in [10]:
                ax2.plot(x, reduced_sol_rounds, marker =reduce_markers[i], color = reduce_colors[i], linestyle = '-')
    legend_labels = []  # 存储图例标签
    legend_handles = []  # 存储图例句柄
    # 添加关于 difficulty 的图例
    for j, difficulty in enumerate(sorted(set(k[1] for k in data_dict.keys() if k[1] in difficulties))):
        legend_labels.append(f"difficulty={difficulty}")
        legend_handles.append(plt.Line2D([0], [0], marker=markers[j], color=colors[j], label=f"{difficulty}"))

    # 添加关于 var_num 的图例
    for i, var_num in enumerate(sorted(set(k[0] for k in data_dict.keys()))):
        legend_labels.append(f"var_num={var_num}")
        legend_handles.append(plt.Line2D([0], [0], color='gray', linestyle=linestyles[i], linewidth=2))

    # plt.ylim(min_data-20, max_data+20)
    ax1.set_ylim(ylim[0],ylim[1])
    x_ticks = np.arange(0, 20 + 1, 1)
    ax1.set_xticks(x_ticks)
    ax1.set_xlabel('Number of miners')
    ax1.set_ylabel(ylabel)
    ax1.legend(legend_handles, legend_labels, loc='best')
    ax1.grid()
    if entry == 'ave_solve_round':
        # 创建椭圆并添加到图中
        ellipse_subunpub = Ellipse((6, 550), width=0.3, height=120, edgecolor='#ff8c00', facecolor='none', linestyle='--', linewidth=2)

        # 添加文本框和箭头
        ax2.add_patch(ellipse_subunpub)
        ax2.annotate("Time reduction \nw/ var_num = 100",
                     xy=(9, 0.53),
                     xytext=(-100, 48),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color='#0072BD', linewidth=2),
                     color='#0072BD',
                     fontsize = 16,)

        ax2.annotate("Time reduction w/ var_num = 50",
                     xy=(14, 0.46),
                     xytext=(-100, 20),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color='#b22222', linewidth=2),
                     color='#b22222',
                     fontsize = 16,)
        # ax1.spines['left'].set_color('#00796B')  # Set the color of the y-axis to blue
        # ax1.yaxis.label.set_color('#00796B')
        # ax1.tick_params(axis='y', colors='#00796B')
        # ax2.yaxis.label.set_color('#ff8c00')
        # ax2.tick_params(axis='y', colors='#ff8c00')
        # ax2.spines['right'].set_color('#ff8c00')  # Set the color of the second y-axis to green
        ax2.set_ylim(0,1)
        ax2.set_ylabel("The ratio of time reduction in solving with difficulty 10")
    plt.show()
    plt.close()

def plot_line_chart_attaker(data_dict, ylabel, ylim, entry):
    """
    {(dmin, adversary_num):difficulty:{data}}
    data_dict格式实例: 
    {(1, 5): 
        {10: {'ave_mb_forkrate': 0.4166666666666667,
            'ave_solve_round': 33.6,
            'ave_subpair_num': 13.11111111111111,
            'total_mb_forkrate': 0.46},
        15: {'ave_mb_forkrate': 0.40626984126984117,
            'ave_solve_round': 29.7,
            'ave_subpair_num': 12.0,
            'total_mb_forkrate': 0.42},
        20: {'ave_mb_forkrate': 0.311038961038961,
            'ave_solve_round': 27.3,
            'ave_subpair_num': 14.222222222222221,
            'total_mb_forkrate': 0.47368421052631576}}
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小
    x_range = []
    legend_list = []
    dmins = []
    adv_nums = []
    data_groups = defaultdict(dict)

    
    for (dmin, adv_num), data_line in data_dict.items():
        if dmin!=4 and (adv_num == 1 or adv_num == 3 or adv_num == 5 or adv_num == 8):
        # if dmin == 2:
            dmins.append(dmin)
            adv_nums.append(adv_num)
            # legend_list.append(f"dmin = {int(dmin)}, adversary_num = {int(adv_num)}")
            x_range = []
            entry_data_line = []
            for miner_num, data_point in data_line.items():
                x_range.append(int(miner_num))
                entry_data_line.append(data_point[entry])
            data_groups[dmin][adv_num] = entry_data_line

    max_data = 0
    min_data = 200
    colors = [
        '#0072BD','#ffa500','#32cd32','#b22222','#9467bd', '#87cefa', '#ff6347',
        '#bcbd22','#00b1bd','#D95319', '#cc6600','#00008b', '#000000','#ff8c00',
        '#f08080','#1f77b4','#ff7f0e','#2ca02c','#d62728', '#9467bd']
    print(legend_list)
    fig = plt.figure(figsize=(10, 6.5))
    linestyles = ['-','--','-.']
    markers = ['o','*','^']
    plts = []
    for group_idx, (dmin, data_lines) in enumerate(data_groups.items()):
        for plt_index, (adv_num, data) in enumerate(data_lines.items()):
            l1, = plt.plot(
                x_range, 
                data, 
                label=f"dmin = {int(dmin)}, adversary_num = {int(adv_num)}", 
                linestyle = linestyles[group_idx], 
                marker = markers[group_idx], 
                color = colors[plt_index]
            )
            plts.append(l1)
            max_data = max(data) if max(data)>max_data else max_data
            min_data = min(data) if min(data)<min_data else min_data
    
    plt.ylim(ylim[0],ylim[1])
    plt.xlim(x_range[0]-0.5,x_range[len(x_range)-1]+0.5)
    ax = fig.gca()
    plt.xlabel('Difficulty level')
    plt.ylabel(ylabel)
    ax.legend(handles = plts, loc='best')
    ax.grid()
    plt.show()
    plt.close()

def plot_bar_chart_attack(data_dict, ylabel, entry, ylim):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    # plt.rcParams['mathtext.fontset'] = 'custom'
    # plt.rcParams['mathtext.rm'] = 'Times New Roman'
    # plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    # plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    # plt.rcParams['text.usetex'] =  True
    
    plt.rcParams['font.size'] = 16  # 调整字号大小

    # 过滤出选定的adversary_num的数据
    filtered_data = {key: value for key, value in data_dict.items() if key[1] in [1]}
    
    # 获取所有的difficulty和dmin
    difficulties = sorted(list(set([inner_key for value in filtered_data.values() for inner_key in value.keys()])))
    atk_thres = sorted(list(set([key[0] for key in filtered_data.keys()])), reverse=True)
    
    # 设置柱状图的宽度和位置
    bar_width = 0.15
    index = range(len(atk_thres))
    fig= plt.figure(figsize=(10, 6.5))
    # 设置字体为Times New Roman
    colormap = plt.cm.get_cmap('Oranges', len(difficulties)+1)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小s
    
    # 对于每个dmin，绘制一个柱形
    for i, difficulty in enumerate(difficulties):
        values = [filtered_data.get((atk_thre, 1)).get(difficulty).get(entry) for atk_thre in atk_thres]
        
        bars = plt.bar([ind + i * bar_width for ind in index], 
                        values, width=bar_width, 
                        label=f'difficulty = {difficulty}',color=colormap(i+1))
    
    plt.xlabel('Safe thresholds on the theory success rate of plagrism attack ', fontfamily='Times New Roman')
    plt.ylabel(ylabel, fontfamily='Times New Roman')   
    # 调整xticks的位置，使其位于每组条形图的中间
    plt.xticks([ind + bar_width * ((len(difficulties) -1)/ 2) for ind in index], 
               [str(atk_thre) for atk_thre in atk_thres], 
               fontfamily='Times New Roman')
    
    # plt.yscale('log')
    ax = fig.gca()
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    # ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[0.5]))
    # ax.yaxis.set_major_formatter(plt.FuncFormatter())
    # def minor_formatter(x, pos):
    # # 将次要坐标轴的值转换为所需的格式
    #     return f'${int(x)} \\times 10^{{-4}}$'

    # ax.yaxis.set_minor_formatter(FuncFormatter(minor_formatter))
    # ax.yaxis.set_minor_formatter(ScalarFormatter(useMathText=True))
    
    plt.legend()
    plt.tight_layout()
    plt.grid(which='both')
    plt.show()

def plot_stacked_bar_chart(data_dict, ylabel, entry):
    difficulties = sorted(list(set([inner_key for value in data_dict.values() for inner_key in value.keys()])))
    safe_thres = sorted(list(set([key[0] for key in data_dict.keys()])))
    adv_nums = sorted(list(set([key[1] for key in data_dict.keys() if key[1] in [1]])))
    
    adv_colormaps = ['Oranges', 'Blues', 'Greens', 'Reds','Purples', 'Greys']
    adv_num_colors = ['#0072BD','#ffa500','#32cd32','#ff6347','#b22222','#9467bd', '#87cefa']
    # adv_num_colors = sns.color_palette("muted", len(adv_nums))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小s

    fig,ax = plt.subplots(figsize=(10, 6.5))
    bar_width = 0.2
    for adv_index, adv_num in enumerate(adv_nums):
        colormap = plt.cm.get_cmap(adv_colormaps[adv_index], len(safe_thres)+1)
        
        for difficulty_index, difficulty in enumerate(difficulties):
            bottom_value = 0
            max_height = 0
            for dmin in safe_thres:
                value = data_dict.get((dmin, adv_num), {}).get(difficulty, {}).get(entry, 0) - bottom_value
                bar = ax.bar(
                    difficulty_index + adv_index * 0.2, 
                    value, 
                    width=bar_width, 
                    bottom=bottom_value, 
                    color=colormap(len(safe_thres) - 1 - safe_thres.index(dmin)),
                )
                bottom_value += value
                max_height = max(max_height, bottom_value)
            # if bottom_value == 0:
            #     for last_bar in last_bars:
            #         yval = last_bar.get_height()
            #         plt.text(last_bar.get_x() + last_bar.get_width()/2, 
            #         yval + 0.01, 
            #         round(yval, 2), 
            #         ha='center', 
            #         va='bottom', 
            #         fontfamily='Times New Roman')
            # Draw rectangle around the stacked bar
            rect = plt.Rectangle(
                (bar[0].get_x(), 0), 
                bar_width, 
                max_height, 
                linewidth=0.5, 
                edgecolor='lightgray', 
                facecolor='none'
            )
            ax.add_patch(rect)
    index = range(len(difficulties))
    print([ind + bar_width * ((len(safe_thres) -1)/ 2) for ind in index])
    plt.xticks([ind + bar_width * ((len(safe_thres) -1)/ 2) for ind in index], 
            [str(difficulty) for difficulty in difficulties], 
            fontfamily='Times New Roman')
    ax = fig.gca()
    ax.set_xlabel('Difficulty')
    ax.set_ylabel(ylabel)
    ax.set_ylim([0,0.002])
    # Create custom legend for dmins and adv_nums
    adv_num_legend_elements = [Patch(facecolor=plt.cm.get_cmap(adv_colormaps[i])(0.5), 
        label=f'adv_num={adv_num}') for i, adv_num in enumerate(adv_nums)]
    dmin_legend_elements = [Patch(facecolor='grey', alpha=1-(i)/len(safe_thres), 
        label=f'safe threshold={safe_thre}') for i, safe_thre in enumerate(safe_thres)]
    ax.legend(handles=adv_num_legend_elements+dmin_legend_elements, loc='upper right')
    ax.grid()
    ax.set_title(ylabel)
    plt.tight_layout()
    plt.show()

def plot_bar_chart_kb_forkrate(data_dict, ylabel):
    """
        data_dict格式 {(strategy, difficulty):data}
        ylabel: 'Key-Block Fork Rate', 'Mini-Block Stale Rate'
    """
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

def plot_bar_chart_diff_search(data_dict, key, ylabel):
    """
        data_dict格式 {(strategy, difficulty):data}
        ylabel: 'Key-Block Fork Rate', 'Mini-Block Stale Rate'
    """
    # 设置颜色方案为 "Set2"

    # 设置字体为 Times New Roman，字号为12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小

    strategy_order = {
        "random": 0,
        "breath first w. mini-block": 1,
        "deep first w. mini-block": 2,
        "best first w. problem": 3,
    }
    def sort_strategies(strategy):
        return strategy_order.get(strategy, len(strategy_order))

    # 从输入字典中提取数据
    miner_nums = sorted(list(set(key[1] for key in data_dict.keys())))
    strategies = sorted(list(set(key[0] for key in data_dict.keys())), key=sort_strategies)
    data = np.array(
        [[data_dict[(strategy, difficulty)][key] for strategy in strategies] 
        for difficulty in miner_nums])
    colors = plt.cm.get_cmap('Greens', len(strategies)+1)

    # 绘图
    fig= plt.figure(figsize=(10, 6.5))
    bar_width = 0.2
    x_ticks = np.arange(len(miner_nums))

    for i, strategy in enumerate(strategies):
        plt.bar(x_ticks + i * bar_width, 
               data[:, i], 
               bar_width, 
               label=strategy, 
               color=colors(i+1))
    ax = fig.gca()
    ax.set_xlabel('Number of miners')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_ticks + bar_width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(miner_nums)
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()

def draw_radars(data_dict):
    def draw_radar_chart(grouped_data:pd.DataFrame, title,miner_num):
        # 设置字体为 Times New Roman，字号为12
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 16  # 调整字号大小
        entries = grouped_data.columns.tolist()
        N = len(entries)

        angles = [(n / float(N) * 2 * math.pi + math.pi/2) % (2 * math.pi) for n in range(N)]
        angles += angles[:1]
        fig, ax = plt.subplots(figsize=(10, 6.5), subplot_kw=dict(polar=True))
        
        plt.xticks(angles[:-1], entries)
        # 调整0度和180度的标签位置
        # 隐藏特定角度的原有标签并使用text函数添加新标签
        for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
            if angle==0 or angle== 2 * math.pi or angle==math.pi:
                label.set_visible(False)
                if angle == 0 or angle == 2 * math.pi:
                    y_offset = 0.2
                else:  # angle == math.pi
                    y_offset = 0.1
                ax.text(angle, 1.1+y_offset, label.get_text(), 
                        horizontalalignment='center',
                        verticalalignment="center", 
                        transform=ax.get_xaxis_transform(), 
                        fontsize=14)

        ax.set_rlabel_position(180)
        plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
        plt.ylim(0, 1)

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        color_index = 0

        for strategy, st_data in grouped_data.iterrows():
            stats = st_data.to_list()
            stats += stats[:1]
            current_color = colors[color_index % len(colors)]
            ax.plot(angles, stats, color=current_color, linewidth=1.5, linestyle='solid', label=strategy)
            ax.fill(angles, stats, color=current_color, alpha=0.1)
            color_index += 1

        plt.legend(loc='upper left', bbox_to_anchor=(0.9, 1))
        plt.title(title)
        # ax.set_rasterized(True)
        if SAVE:
            plt.savefig(SAVE_PREFIX + f"\\SearchRadarM{miner_num}.pdf")
        plt.show()
        plt.close()

    
    df = pd.DataFrame(data_dict)
    grouped = df.groupby(['miner_num', 'strategy']).mean()[
        ['Average Solving Rounds', 'Mini-block\nForkrate', 
         'Effective Workload', 'Workload\nwith Wasted',]]
        # 选择要归一化的列
    grouped_max = df.groupby('miner_num')[[
        'Average Solving Rounds', 'Mini-block\nForkrate', 
        'Effective Workload', 'Workload\nwith Wasted']].max()
    
    normalized_grouped = grouped.copy()
    for column in grouped.columns:
        # 对数变换
        # group_max =grouped[column].max()
        for miner_num in grouped_max.index:
            max_value = grouped_max.loc[miner_num, column]
            normalized_grouped.loc[(normalized_grouped.index.get_level_values('miner_num') == miner_num), column] /= max_value

    for miner_num, sts_data in normalized_grouped.groupby(level=0):
        draw_radar_chart(sts_data.droplevel(0), 
            f'Properties of different searching strategies with {miner_num} miner(s)', miner_num)

def plot_atklog_fig6(atklog_mb:list, ax_inset:plt.Axes):
    """
    {"depth":0,"theory":0,"attack_num":0,"success_num":0,"success_rate":0}
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 24  # 调整字号大小
    attacks = range(len(atklog_mb))
    success_rates = [atk['success_rate'] if atk['success_rate']!=0 else 0 for atk in atklog_mb]
    theory_values = [atk['theory'] for atk in atklog_mb]
    lbs = [math.log(1/(50**atk['depth'])) for atk in atklog_mb]

    # fig = plt.figure(figsize=(10, 6.5))
    # ax_inset.bar(attacks, success_rates, label='simulation', color='orange',  width=1, alpha=0.7)
    ax_inset.fill_between(attacks, success_rates, color='orange', alpha=0.5, label='simulation',edgecolor='none')
    ax_inset.plot(attacks, theory_values, label='theory ', color='#1f77b4', linestyle = "--",linewidth = 2)
    # plt.plot(attacks, lbs, label='lowerbound', color='green', alpha=0.7, linestyle = "--")
    # plt.axhline(y=math.log(1/(50**3.5)), label='Lowerbound', color="green", linestyle='--')
    ax_inset.axhline(y=0.001, label='threshold', color="red", linestyle='-.',linewidth = 2)
    # plt.ylim([-17.0,-6.0])
    ax_inset.set_xlim([0, len(atklog_mb)+10])
    ax_inset.set_yscale("log")
    ax_inset.set_xlabel('Blocks')
    ax_inset.set_ylabel('Success rate')
    ax_inset.legend(loc = "lower left")
                    # , bbox_to_anchor=(1, 0.97))
    # plt.legend(loc = "best")
    ax_inset.grid()
    # ax.set_rasterized(True)
    # if SAVE:
    #     plt.savefig(SAVE_PREFIX + "\\atklogm10_001.svg", dpi=300)
    # plt.show()

def plot_security_fig6(atklog):
    data_path = pathlib.Path(".\Result_Data\\1029attack_data2lite.json")
    data_list = []
    with open(data_path, 'r') as f:
        json_list = f.read().split('\n')[:-1]
        for json_data in json_list:
            data_list.append(json.loads(json_data))

    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    df = pd.DataFrame([data for data in data_list if data["difficulty"] in [3,5,8,10]])
    df_sorted = df.sort_values(by=['safe_thre', 'difficulty'],ascending = [False,True])
    sorted_safe_thre = df_sorted['safe_thre'].unique()
    print(sorted_safe_thre)
    print(df)
    np.random.seed(0)  # 为了可重复性的示例
    sns.set(style="whitegrid", font='Times New Roman', font_scale=1.2)
    colors = ["#FF8283", "#0D898A","#f9cc52","#5494CE", ]

    # 创建图表
    heights = [1, 3, 1, 1]
    fig, (ax4, ax1, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 8), 
                                        gridspec_kw={'height_ratios': heights, 'hspace': 0})
    ax4: plt.Axes 
    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes
    
    # 第一个图表：Rate的柱状图
    sns.barplot(x='safe_thre', y='ave_advrate', hue='difficulty', 
                palette = colors, data=df, ax=ax1, width = 0.7, order=sorted_safe_thre)
    ax1.set_xlabel('')  # 移除x轴标签，因为将与第二个图共享
    ax1.set_ylabel('Success Rate')
    ax1.legend(title='Difficulty', loc = "upper left", bbox_to_anchor=(0.15, 0.98))
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.grid(True)

    # 第二个图表：Prob的柱状图
    sns.barplot(x='safe_thre', y='ave_accept_advrate', hue='difficulty', 
                palette = colors, data=df, ax=ax2, width = 0.7,order=sorted_safe_thre)
    ax2.set_ylabel('Chain quality')
    ax2.set_ylim(bottom=0 + 0.00001)
    ax2.get_legend().remove()
    ax2.invert_yaxis()
    # ax2.grid(True)

    # 调整子图间距并显示图表

    # 第三个图表：Wasted的箱型图
    sns.boxplot(x='safe_thre', y="ave_subpair_unpubs", data = df[df["difficulty"] == 5], 
                ax=ax3, width = 0.2, linewidth = 2,order=sorted_safe_thre)
    ax3.set_ylabel('Wasted workload')
    ax3.set_xlabel('Safe threshold')
    ax3.grid(True)
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
    ax4.plot("safe_thre", 'safe_thre' , 
                data = df[df["difficulty"] == 5].sort_values(by="safe_thre"),  
                marker='x',linestyle = "--", color = "#0D898A",
                label = "threshlod")
    ax4.plot("safe_thre", 'ave_advrate' , 
                data = df[df["difficulty"] == 5].sort_values(by="safe_thre"),  
                marker='o',color = "#0D898A",
                label = "simulation")
    ax4.set_ylim(bottom=0, top=0.005)
    ax4.set_xticks(sorted_safe_thre)
    
    print(ax4.get_xlim())
    print(ax3.get_xlim())
    ax4.invert_xaxis()
    ax4.legend(loc = "upper left",bbox_to_anchor=(0.35, 1.05),ncol=2)
    ax4_2 = ax4.twinx()
    ax4_2.plot("safe_thre", 'safe_ratio' , 
                data = df[df["difficulty"] == 5].sort_values(by="safe_thre"), 
                    marker='o', color = "#BC5133")
    ax4_2.spines['left'].set_color('#0D898A')  # Set the color of the y-axis to blue
    ax4_2.set_xticks(sorted_safe_thre)
    # ax_inset.set_yticks([0.00])
    ax4.yaxis.label.set_color('#0D898A')
    ax4.tick_params(axis='y', colors='#0D898A')
    ax4.set_ylabel("Success Rate")
    ax4.set_xlabel("Safe threshold")
    ax4_2.spines['right'].set_color('#BC5133')  # Set the color of the y-axis to blue
    ax4_2.yaxis.label.set_color('#BC5133')
    ax4_2.tick_params(axis='y', colors='#BC5133')
    ax4_2.set_ylabel("Ratio of simulation \nto threshold")
    # ax_inset.grid(False)
    ax4_2.grid(False)
    # 调整子图间距
    ax_inset = ax1.inset_axes([0.5, 0.3, 0.5, 0.65])
    ax1.add_patch(Rectangle((5.6, 0.00001), 0.3, 0.0001, color='#45636A', fill=False, lw=1.5))
    ax1.add_artist(ConnectionPatch(xyA=(5.9, 0.00011), xyB=(6.5, 0.000572), coordsA='data', coordsB='data',
                       axesA=ax1, axesB=ax1, color='#45636A',lw = 0.5))
    ax1.add_artist(ConnectionPatch(xyA=(5.6, 0.00011), xyB=(3, 0.000572), coordsA='data', coordsB='data',
                        axesA=ax1, axesB=ax1, color='#45636A',lw = 0.5))
    
    plot_atklog_fig6(atklog, ax_inset)
    plt.show()
    
def plot_mbtime_grow_fig5(data_df:pd.DataFrame):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14
    fig = plt.figure(figsize=(12, 9))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.5, 1.5, 2],width_ratios=[1.5, 1, 1.5,1]) # 3 rows, 2 columns
    # sns.set(style="whitegrid")
    # colors = ["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"]
    colors = ["#FF8283", "#0D898A","#f9cc52","#5494CE", ]

    def plot_means(ax:plt.Axes, data_df):
        # Unique miners for different lines
        
        miners = data_df['miner_num'].unique()
        for i , miner in enumerate(miners):
            miner_data = data_df[data_df['miner_num'] == miner]
            # ax.errorbar(miner_data['difficulty'], miner_data['mean'], yerr=miner_data['std'], label=miner)
            ax.plot(miner_data['difficulty'], miner_data['mean'],  label=miner, marker = 'o',color = colors[i])

        # ax.set_title(title)
        ax.set_xlabel('Difficulty')
        ax.set_ylabel('Means of block times')
        ax.legend(title="miner num",fontsize = 12)
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5)
    
    def plot_violin(ax:plt.Axes, data_df:pd.DataFrame):
        sns.violinplot(x='mb_times',y='difficulty', hue='miner_num', inner="box", 
                       data=data_df, ax=ax, orient='h',palette = colors[:3], linewidth = 0.1,alpha = 0.1)
        # sns.pointplot(x='mb_times', y='difficulty', data=data_df, hue='miner_num',
            #   orient='h', join=False, markers='D', palette='dark', ci=None, estimator='mean')
        ax.set_xlim([0,100])
        # ax.set_ylabel('Difficulty', labelpad = -10, loc='top')
        ax.legend(title="miner num",fontsize = 12)
        ax.set_ylabel('Difficulty')
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
        # sns.lineplot()
        # c = plt.cm.Greens
        # point_norm = mcolors.Normalize(vmin=0, vmax=len(data_df['miner_num'].unique()))
        # i=1
        data_df.copy().reset_index()
        c = ["#95d5b2", "#74c69d", "#52b788","#40916c"]
        for i,data in data_df.iterrows():
            print(data_df)
            ax.hlines(1/data['ave_mb_growth'], 0, 5000,color=c[i-4],linestyles='--',linewidth = 1)
            # data = data_df[data_df['difficulty'] == d]
            ax.plot(range(len(list(data['grow_proc']))), list([1/d for d in data['grow_proc']]), 
                    label = data['difficulty'],c=c[i-4])
            # i+=1
        ax.set_xlim([0,3000])
        ax.legend(title="difficuty",fontsize = 12)
        ax.set_xlabel('Block')
        ax.set_ylabel('Growth rate')
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5)

    
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
    axGrowth = fig.add_subplot(grid[2, 0:2]) # Placeholder for ax_d (e in the description)
    axGrowProc = fig.add_subplot(grid[2, 2:4])
    axViolin = fig.add_subplot(grid[0:2, 3])

    plot_means(axMeans, data_df)
    plot_violin(axViolin, mb_times_df)
    plot_growthrate(axGrowth, data_df)
    plot_block_time_fig5(axTimesM1, data_df[data_df['miner_num'] == 1])
    plot_block_time_fig5(axTimesM3, data_df[data_df['miner_num'] == 3])
    plot_grow_proc(axGrowProc, data_df[(data_df['miner_num'] == 3) & (data_df['difficulty'].isin([3, 5, 7, 9]))].copy())
    
    ax_list = [axMeans, axTimesM1, axTimesM3, axViolin,axGrowth, axGrowProc]
   
    
    for ax in ax_list:
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')  # 设置为浅灰色
        # ax.text(-0.05, 1, label, transform=ax.transAxes, fontsize=16,
        #     verticalalignment='top', horizontalalignment='left',fontweight='bold')
        # # fig.text(0.1, 0.9, label, fontsize=16, fontweight='bold')
            
    subplot_labels = ['a', 'b', 'c', 'd', 'e', 'f']
    # 假设这些是你想要的编号位置，根据你的图形尺寸进行适当调整
    label_positions = [(0.05, 0.97), (0.3, 0.97), (0.3, 0.67),(0.8, 0.97), 
                       (0.05, 0.37), (0.5, 0.37)]

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

def plot_block_time_pdf(ax:plt.Axes, mb_times_list, difficulties, xlabel):
    """"
    区块时间pdf，并标注峰值与均值
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14
    
    hist_list = []
    bins_list = [] 
    peaks_list = [] # 存储峰值
    prominences_list = [] # 峰值突出度
    avg_values = []  # 存储平均值
    for mb_times in mb_times_list:
        data = np.array(mb_times)
        # 计算频率
        # hist, bins = np.histogram(
        #     data, bins=300, density=True)
        hist, bins = np.histogram(
            data, bins=np.arange(min(data), max(data) + 2), density=True)
        hist_list.append(hist)
        bins_list.append(bins)
        # 找到直方图的峰值和突出度
        peaks, _ = find_peaks(hist)
        prominences = peak_prominences(hist, peaks)[0]
        prominence_threshold = 0.01
        peaks_list.append(peaks)
        prominences_list.append(prominences)
        # 计算平均值
        avg = np.mean(data)
        avg_values.append(avg)
    colors = ['#1f77b4', '#32cd32', '#bcbd22', '#ffa500','#b22222',
            '#9467bd', '#87cefa', '#ff6347',]
    colors = ["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"]
    # colors = ["r","b","y","g","#BEA9E9"]
    # 绘图
    # fig, ax= plt.subplots(figsize=(10, 6.5))
    for plt_idx,(mb_times,hist,bins,peaks,prominences, avg) in enumerate(
            zip(mb_times_list,hist_list, bins_list, peaks_list, prominences_list,avg_values)):
        # ax.bar(bins[:-1], hist, width=1, align='edge', alpha=0.5, color=colors[plt_idx])
        ax.hist(mb_times, bins=np.arange(min(mb_times), max(mb_times) + 2), density=True, 
                alpha=0.3,histtype='stepfilled', edgecolor = "black",linewidth=1.5,color=colors[plt_idx],align="left",zorder = 4-plt_idx)
        # ax.plot(bins[:-1], hist, color=colors[plt_idx],linewidth = 2,zorder = 5)
        # # 标记峰值位置
        # for _, (peak_x,prominence) in enumerate(zip(bins[peaks], prominences)):
        #     if prominence > prominence_threshold:
        #         peak_y = hist[peak_x-1]
        #         # plt.axvline(x=peak_x, color=colors[plt_idx], linestyle='--')
        #         ax.scatter(peak_x, peak_y, marker='*', color='red', zorder=3)
        #         arrowprops = dict(arrowstyle='->', color=colors[plt_idx])
        #         ax.annotate(f'{peak_x:.2f}', 
        #                     xy=(peak_x, peak_y), 
        #                     xytext=(0+5*plt_idx, 70-plt_idx*10.5), 
        #                     textcoords='offset points', 
        #                     arrowprops=arrowprops, 
        #                     color=colors[plt_idx])
        # 标记平均值位置
        avg_y = np.interp(avg, bins[:-1], hist)
        # ax.axvline(x=avg, color=colors[plt_idx], linestyle='--',linewidth = 1.5)
        ax.scatter(avg, avg_y, marker='o', color='red', zorder=5)
        arrowprops = dict(arrowstyle='->', color=colors[plt_idx])
        ax.annotate(f'{avg:.2f}', 
                    weight='bold',
                    xy=(avg+0.005, avg_y+0.0001), 
                    xytext=(20+plt_idx*10, 30 + plt_idx*20), 
                    textcoords='offset points',
                    arrowprops=arrowprops, 
                    color=colors[plt_idx])
    # 添加 "Averages" 标签
    ax.annotate('Means', 
                xy=(13, 0.06), 
                xytext=(0, 0), 
                textcoords='offset points',color='red',weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', 
                          ec='k',lw=1))
    
    # # 添加 "Peaks" 标签
    # ax.annotate('Peaks', xy=(30, 0.03), xytext=(0, 0),
    #              textcoords='offset points', color='red', zorder=5,weight='bold',
    #              bbox=dict(boxstyle='round,pad=0.5', fc='yellow', 
    #                        ec='k',lw=1 ,alpha=0.4))
    lgdhdls = [ax.plot([],[],color=colors[i],label=d)[0] for i, d in enumerate(difficulties)]

    # lgdhdls.append(ax.scatter(-100,-100,color="red",label="peaks",marker="*"))
    lgdhdls.append(ax.scatter(-100,-100,color="red",label="means",marker="o"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability mass function')
    x_tick_step = 5
    x_ticks = np.arange(0, max(data) + x_tick_step, x_tick_step)
    x_ticks = np.insert(x_ticks, 0, min(data))
    ax.set_xticks(x_ticks)
    # ax.set_xlim(min(data), max(data)+1)
    ax.set_xlim(min(data), 50+1)
    ax.set_ylim(0,0.15)
    ax.legend(handles=lgdhdls,loc='best', title = "difficulty")
    # ax.grid()
    # ax.set_rasterized(True)
    # if SAVE:
    #     plt.savefig(SAVE_PREFIX + "\\mbtimesv100m5.eps", dpi=300)
    # plt.show()

def plot_keyblock_time_pdf(block_time_lists, var_nums, xlabel,
                        x_tick_step, prominence_threshold):
    """"
    区块时间pdf，并标注峰值与均值
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小
    hist_list = []
    bins_list = [] 
    kde_ylist = []
    kde_xlist = []
    peaks_list = [] # 存储峰值
    # prominences_list = [] # 峰值突出度
    for block_time_list in block_time_lists:
        data = np.array(block_time_list)
        # 计算频率
        hist, bins = np.histogram(data, bins=500, density=True)
            # data, bins=np.arange(min(data), max(data) + 2), density=True)
        hist_list.append(hist)
        bins_list.append(bins)
        # 核密度估计
        kde = gaussian_kde(data, bw_method=0.1)
        kde_x = np.linspace(min(data), max(data), 1000)
        kde_y = kde(kde_x)
        kde_xlist.append(kde_x)
        kde_ylist.append(kde_y)
        # # 找到核密度估计的峰值位置
        # peaks, _ = find_peaks(kde_y)
        # prominences = peak_prominences(kde_y, peaks)[0]
        # prominence_threshold = 0.001
    colors = ['#1f77b4','#ffa500','#32cd32','#b22222',
            '#9467bd', '#87cefa', '#ff6347','#bcbd22']
    # 绘图
    fig,ax = plt.subplots(figsize=(15, 3))
    plts = []
    for plt_idx,(hist,bins, kde_y, kde_x) in enumerate(
            zip(hist_list, bins_list, kde_ylist, kde_xlist)):
        print(kde_x)
        ax.plot(bins[:-1], hist, color=colors[plt_idx], alpha = 0.5)
        # ax.hist(bins[:-1],bins, color=colors[plt_idx], alpha = 0.5)
        l1, = ax.plot(
            kde_x, 
            kde_y, 
            color=colors[plt_idx], 
            label = f"var_num = {var_nums[plt_idx]}", 
            linewidth=2)
        plts.append(l1)
    
    ax = fig.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability mass function')
    x_ticks = np.arange(0, max(data) + x_tick_step, x_tick_step)
    x_ticks = np.insert(x_ticks, 0, min([min(t) for t in block_time_lists]))
    ax.set_xticks(x_ticks)
    ax.set_xlim(min([min(t) for t in block_time_lists]), 1500)
                # max([max(t) for t in block_time_lists])+1)
    ax.set_ylim(0, 0.05)
    ax.legend(handles = plts, loc='best')
    ax.grid()
    plt.show()

def verify_mbtimes(block_time_lists, difficulties, xlabel,
                   x_tick_step, prominence_threshold):
    """"
    区块时间pdf，并标注峰值与均值
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 14  # 调整字号大小
    hist_list = []
    bins_list = [] 
    peaks_list = [] # 存储峰值
    prominences_list = [] # 峰值突出度
    avg_values = []  # 存储平均值
    for block_time_list in block_time_lists:
        data = np.array(block_time_list)
        # 计算频率
        hist, bins = np.histogram(
            data, bins=np.arange(min(data), max(data) + 2), density=True)
        hist_list.append(hist)
        bins_list.append(bins)
        # 找到直方图的峰值和突出度
        peaks, _ = find_peaks(hist)
        prominences = peak_prominences(hist, peaks)[0]
        prominence_threshold = 0.01
        peaks_list.append(peaks)
        prominences_list.append(prominences)
        # 计算平均值
        avg = np.mean(data)
        avg_values.append(avg)
    colors = ['#1f77b4', '#32cd32', '#bcbd22', '#ffa500','#b22222',
            '#9467bd', '#87cefa', '#ff6347',]
    # 绘图
    fig, ax= plt.subplots(figsize=(10, 6.5))
    for plt_idx,(hist,bins,peaks,prominences, avg) in enumerate(
            zip(hist_list, bins_list, peaks_list, prominences_list,avg_values)):
        # ax.bar(bins[:-1], hist, width=1, align='edge', alpha=0.7, color='#1f77b4')
        ax.plot(bins[:-1], hist, color=colors[plt_idx])
    lgdhdls = [ax.plot([],[],color=colors[i],label=f"difficulty = {d}")[0] for i, d in enumerate(difficulties)]
    p =0.1
    m = 2
    k_range = 1000
    print("miner_num: ", m)
    print("prblm_num: ", 3)
    print("prob: ", p)
    
    # theroy1 = [2*p*((1-p)**(k-1))*(1-(1-p)**(k-1))+p*p*(1-p)**(2*k-2) for k in range(1,k_range)]
    theroy1 = [((1-p)**(k-1))*(p) for k in range(1,k_range)]
    theroy1.insert(0,0)
    theroy2 = np.convolve(theroy1,theroy1, mode="full")
    theroy2 = theroy2 / np.sum(theroy2)
    theroy3 = np.convolve(theroy2[:len(theroy1)],theroy1, mode="full")
    theroy3 = theroy3 / np.sum(theroy3)
    theroy = theroy3
    mu_theroy = []
    unpubs=[]
    unnums=[]
    fks=[]
    for k in range(0,len(theroy1)):
        thk = 0
        for i in range(0, k+1):
            thk += theroy[i-1]
        th = 0
        for i in range(1,m+1):
            th += math.comb(m, i)*(1-thk)**(m-i) * theroy[k-1]** i
        fk = 0
        for i in range(2,m+1):
            fk += math.comb(m, i)*(1-thk)**(m-i) * theroy[k-1]** i
        fks.append(fk)
        unpubk = 0
        for i in range(1,k+1):
            pni1 = theroy1[i-1]
            # pni2= theroy2[i-1]
            for j in range(i+1,k+1):
                pni1 *= (1-p)
                # pni2 *= (1-p)
            unpubk += (pni1)
        unpubs.append(unpubk*theroy[k-1])
        # unpubs.append((k)*theroy1[k-1]*theroy2[k-1])
        # unpubk1 = 0
        # unpubk2 = 0
        # unpubk3 = 0
        # for i in range(1,k+1):
        #     unpubk1 += theroy1[i]
        # for i in range(2,k+1):
        #     unpubk2 += theroy2[i]
        # for i in range(3,k+1):
        #     unpubk3 += theroy3[i]
        # unpubs.append((unpubk1*(1-unpubk2)+unpubk2*(1-unpubk3))*theroy3[k])
        # unnums.append((unpubk1*(1-unpubk2)+2*unpubk2*(1-unpubk3))*theroy3[k])
        # unpubs.append(unpubk1*(1-unpubk2)*theroy[k])
        # print(str(pn)+'\n')
        # mu_theroy.append(th)
    mu_theroy.insert(0,0)
    unpubs.insert(0,0)
    fks.insert(0,0)
    print(f"pnums: {np.sum(unpubs)}")
    unpubs = list(unpubs / np.sum(unpubs))
    print(f"fk: {np.sum(fks)}" )
    fks = list(fks / sum(fks))
    
    ax.plot([i for i in range(0,k_range-1)], mu_theroy[1:k_range], color="orange", linestyle='--', marker = "+")
    ax.plot([i for i in range(0,k_range-1)], unpubs[1:k_range], color="red", linestyle='--', marker = "+")
    ax.plot([i for i in range(0,k_range-1)], fks[1:k_range], color="black", linestyle='--', marker = "+")
    lgdhdls.append(ax.plot([],[],color="orange", label="theroy", linestyle='--', marker = "+")[0])

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    x_ticks = np.arange(0, max(data) + x_tick_step, x_tick_step)
    ax.set_xticks(x_ticks)
    ax.set_xlim(0, 20)
    ax.set_ylim(0,0.5)
    ax.legend(handles=lgdhdls,loc='best')
    ax.grid()
    plt.show()


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


def verify_correctness():
    prob =  [0.01, 0.05, 0.1, 0.2, 0.3]
    m2_2pair = {
        "fk_theory": [0.002512616, 0.01282269, 0.026288686, 0.0546414, 0.0828831],
        "fk_simulation": [0.002792182, 0.01290625, 0.0251035827, 0.054969286, 0.08504506]
    }

    # Two pair chain, m=3
    m3_2pair = {
        "fk_theory": [0.0044644219, 0.02279194, 0.047034747, 0.100406469, 0.158018015],
        "fk_simulation": [0.004711, 0.021558, 0.04583187, 0.0942849, 0.1561181]
    }

    # Three pair, m=2
    m2_3pair = {
        "fk_theory": [0.0018863, 0.0096195876, 0.01977244, 0.04195195, 0.067614],
        "fk_simulation": [0.00169711, 0.00914886, 0.0204082, 0.0391698, 0.0645463]
    }

    # Three pair, m=3
    m3_3pair = {
        "fk_theory": [0.003226117, 0.016415361, 0.033634161, 0.07101437, 0.1135875],
        "fk_simulation": [0.003123546, 0.016780283, 0.032975534, 0.0689013, 0.1087344]
    }


    fig = plt.figure(figsize=(12, 8))

    # Plot for Two pair chain, m=2
    plt.plot(prob, m2_2pair['fk_theory'], 
             label='Two pair m=2 (Theory)', 
             linestyle='-', 
             color='blue',
             marker = 'o', 
             )
    plt.plot(prob, m2_2pair['fk_simulation'], 
             label='Two pair m=2 (Simulation)', 
             linestyle='--', color='blue',
             marker = 'o',
             fillstyle='none')

    # Plot for Two pair chain, m=3
    plt.plot(prob, m3_2pair['fk_theory'], 
             label='Two pair m=3 (Theory)', 
             linestyle='-', color='green',
             marker = 'o', )
    plt.plot(prob, m3_2pair['fk_simulation'], 
             label='Two pair m=3 (Simulation)', 
             linestyle='--', color='green',
             marker = 'o',
             fillstyle='none')

    # Plot for Three pair, m=2
    plt.plot(prob, m2_3pair['fk_theory'], 
             label='Three pair m=2 (Theory)', 
             linestyle='-', color='red',
             marker = 'o', )
    plt.plot(prob, m2_3pair['fk_simulation'], 
             label='Three pair m=2 (Simulation)', 
             linestyle='--', color='red',
             marker = 'o',
             fillstyle='none')

    # Plot for Three pair, m=3
    plt.plot(prob, m3_3pair['fk_theory'], 
             label='Three pair m=3 (Theory)', 
             linestyle='-', color='orange',
             marker = 'o', )
    plt.plot(prob, m3_3pair['fk_simulation'], 
             label='Three pair m=3 (Simulation)', 
             linestyle='--', color='orange',
             marker = 'o',
             fillstyle='none')
    ax = fig.gca()
    # ax.plot(diffs, l1, label = "random")
    # ax.plot(diffs, l2, label = "semi-depth-first")
    # ax.plot(diffs, l3, label = "depth-first")
    # ax.plot(diffs, l4, label = "breath-first")
    plt.xlabel('Solving probabilty for a sub-problem pair')
    plt.ylabel('Fork Rate')
    plt.legend()
    ax.set_ylim(0, 0.2)
    plt.grid()
    # plt.show()

    # ax.plot([i for i in range(1,100)], [(0.95**(k-1))*0.05 for k in range(1,100)])
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(12, 8))
    m2_2pair_unpub = {
        "unpub_theory": [1.0050464, 1.0258152, 1.0515472, 1.0973688,  1.13983308],
        "unpub_simulation": [1.00865, 1.0252, 1.04365, 1.060222, 1.13983308]
    }
    m2_3pair_unpub = {
        "unpub_theory": [2.2568628, 2.277366, 2.30297736, 2.3472, 2.37827],
        "unpub_simulation": [2.259203681472589, 2.2755, 2.30725, 2.3492, 2.3778]
    }

    plt.plot(prob, m2_2pair_unpub['unpub_theory'], 
             label='Two pair m=2 (Theory)', 
             linestyle='-', 
             color='blue',
             marker = 'o', 
             )
    plt.plot(prob, m2_2pair_unpub['unpub_simulation'], 
             label='Two pair m=2 (Simulation)', 
             linestyle='--', color='blue',
             marker = 'o',
             fillstyle='none')

    # Plot for Two pair chain, m=3
    plt.plot(prob, m2_3pair_unpub['unpub_theory'], 
             label='Two pair m=3 (Theory)', 
             linestyle='-', color='green',
             marker = 'o', )
    plt.plot(prob, m2_3pair_unpub['unpub_simulation'], 
             label='Two pair chain m=3 (Simulation)', 
             linestyle='--', color='green',
             marker = 'o',
             fillstyle='none')
    plt.xlabel('Solving probabilty for a sub-problem pair')
    plt.ylabel('Number of unpublished sub-problem pairs')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    import json

    import pandas as pd

    # y1_5 = [0, 0.26596617182437166, 0.36544861584069055, 0.47692881506909435, 0.5307483263934047, 0.5590829235010621]
    # y1_10 = [0, 0.2505992743491491, 0.3255964509597138, 0.42869889707874054, 0.4725354529276098, 0.5025691551521008]
    # x1 = [1, 3,5,10,15,20]
    # y2_5 = [0, 0.3404380666473934, 0.44082583972401346, 0.550692167511015,0.6328006717797874]
    # y2_10 = [0, 0.3574133516435062, 0.46108068761020127, 0.5607031585605644, 0.6295804578445527]
    # x2 = [1, 3,5,10,20]
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.plot(x1, y1_5, marker = '+', color = 'b')
    # ax.plot(x1, y1_10, marker = '+',color = 'r')
    # ax.plot(x2,y2_5, marker = 'o', color = 'b')
    # ax.plot(x2,y2_10, marker = 'o',color = 'r')
    # plt.show()
    # file_path = './Result_Data/1210short_data测试不同st.json'
    # # Reading the new file into a DataFrame
    # df = pd.read_json(file_path, lines=True)
    # # Filtering for the required combinations of "openblk_st" and "openprblm_st"
    # filtered_df = df[df['openblk_st'].isin(['openblock_deepfrist', 'openblock_random', 'openblock_breathfirst']) &
    #                     df['openprblm_st'].isin(['openprblm_random', 'openprblm_bestbound'])]
    # # Grouping by 'miner_num', 'openblk_st', and 'openprblm_st', and then calculating the mean of 'total_mb_forkrate' and 'ave_solve_round'
    # grouped_df = filtered_df.groupby(['miner_num', 'openblk_st', 'openprblm_st']).agg({'total_mb_forkrate':'mean', 'ave_solve_round':'mean'}).reset_index()
    # # Pivoting the data for plotting
    # pivot_forkrate = grouped_df.pivot_table(index='miner_num', columns=['openblk_st', 'openprblm_st'], values='total_mb_forkrate')
    # pivot_solveround = grouped_df.pivot_table(index='miner_num', columns=['openblk_st', 'openprblm_st'], values='ave_solve_round')
    # # Plotting the results
    # # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 15))
    # fig,ax1 = plt.subplots(figsize=(10, 6.5))
    # pivot_forkrate.plot(ax=ax1, kind='bar')
    # ax1.set_title('Impact of Strategies on Total MB Fork Rate by Miner Number')
    # ax1.set_ylabel('Total Fork Rate of mini-blocks')
    # ax1.grid(True)
    # plt.show()
    # plt.close()
    # fig,ax2 = plt.subplots(figsize=(10, 6.5))
    # pivot_solveround.plot(ax=ax2, kind='bar')
    # ax2.set_title('Impact of Strategies on Average Solve Round by Miner Number')
    # ax2.set_ylabel('Average Solve Round')
    # ax2.grid(True)
    # plt.show()

        