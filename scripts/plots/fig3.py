import math
from pathlib import Path
import sys
import time
from collections import defaultdict
sys.path.append("E:\Files\gitspace\\bbb-github")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from matplotlib.patches import (
    Rectangle,
)
import json
from matplotlib.gridspec import GridSpec

MAXSAT='maxsat'
TSP='tsp'
MIPLTP='miplib'

SAVE_PREFIX = "E:\Files\A-blockchain\\branchbound\\figs\\fig3"

def plot_bounds_fig3(file_path, type, m=None, ax:plt.Axes = None):
    with open(file_path, 'r') as f:
        jsondata_list = f.read().split('\n')
        data_list = [dict(json.loads(js)) for js in jsondata_list]
    # 读取并解析文件
    # with open(file_path_new, 'r') as file:
    #     json_data_new = json.load(file)
    # 开始绘图
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 20  # 调整字号大小

    if ax is None:
        if type == TSP:
            fig = plt.figure(figsize=(12, 5))
        else:
            fig = plt.figure(figsize=(8, 5))
        ax = fig.gca()

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
            ax.plot([start_point[0], end_point[0], end_point[0]], [start_point[1], start_point[1], end_point[1]], 
                    color = color, linestyle=linestyle, linewidth=linewidth, alpha=0.7, zorder=6)


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
    if type == MAXSAT:
        for point in ub_data:
            if point["ub"] is None:
                continue
            point["ub"] = -point["ub"]
    # if type == TSP:
    #     for point in ub_data:
    #         if point["ub"] is None:
    #             continue
    #         point["ub"] = point["ub"]
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
    children_counts[((0,0),0)]=bpre_df['children_count'].max()
    print(ub_df)
    print(block_df)
    print(root_df)
    print(pre_df)

     # sns.set(style="white")
    
    # 对于相同的整数解，只保留最早出现的点
    int_df = ub_df[(ub_df["allInteger"] == True) & (ub_df["block"]!= "None")].copy()
    int_df = int_df.sort_values('bround')  # 按轮次排序
    int_df = int_df.drop_duplicates(subset=['ub'], keep='first')  # 对于相同的ub值只保留第一次出现的点
    # 标记allInteger路径
    int_paths = []
    intpath_points = set()
    max_bround_point = None
    max_bround = float('-inf')
    for point in ub_data:
        if not ((point['ub'], point['bround']) in int_df[['ub', 'bround']].itertuples(index=False, name=None)):
            continue
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
    for path in int_paths:
        if max_bround_point in path:
            # 用红色标记这条路径
            draw_int_path(path, "red", 5,'-')
        else:
            # 其他路径使用默认颜色
            draw_int_path(path)

    sampled_points = set(root_df['pname']).union(intpath_points).union((((0,0),0)))
    # sampled_points = set(ub_df.sample(frac=0.1, random_state=0)['pname'])
    # sampled_points = sampled_points.union(intpath_points)
    # ["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"] 
    # ["#FF8283", "#0D898A","#f9cc52","#5494CE", ] '#00796B' '#ff8c00' '#b22222'
    # 绘制UB数据点和连接线
    smain = 15
    smain = 15 if type == MAXSAT else 40
    if type == TSP:
        s = 10
    elif type == MAXSAT:
        s = 40
    else:
        s = 15
    sopt = 80 if type == TSP else 40
    rasterized=False if type == MIPLTP else True
    if type != TSP:
        sns.scatterplot(x="bround",y ="ub",
                        data = ub_df[(ub_df["fathomed"] == False) & (ub_df["block"]!= "None")] ,
                        s = smain, color = '#0072BD', rasterized=rasterized ,edgecolor="none",
                        zorder = 5, alpha = 0.7) ,
    sns.scatterplot(x="bround",y ="ub",
                    data = ub_df[(ub_df["fathomed"]== True) & 
                                 (ub_df["allInteger"]==False) & (ub_df["block"]!= "None")] , 
                    color = "#ff8c00", s= s, rasterized=rasterized, edgecolor="none",alpha = 0.5)
    sns.scatterplot(x="round",y ="ub",data = ub_df[(ub_df["block"]== "None")], 
                    color = "#9acd32", s= s,rasterized=rasterized,edgecolor="none",zorder = 4, alpha = 0.5)
    sns.scatterplot(x="bround",y ="ub",data = ub_df[(ub_df["isFork"]== True) & (ub_df["block"]!= "None")] , 
                    color = "black", s= s,rasterized=rasterized,edgecolor="none",zorder = 4, alpha = 0.5)
    
    sns.scatterplot(x="bround",y ="ub",data = int_df, 
                    color = "r", s = sopt,rasterized=rasterized,edgecolor="none",zorder = 6)
    point_norm = mcolors.Normalize(vmin=0, vmax=bpre_df['children_count'].max())
    print(bpre_df['children_count'].max())
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
            print(row)
        # 提取每个block的数据
            blocks.append(row['block'])
            min_ub = row['ub_min']
            max_ub = row['ub_max']
            # 绘制圆角矩形
            width = 0.023 if type == MAXSAT else 0.04
            children_count = children_counts[row['pre_pname']]
            color = blues(0 + 0.4 * (1-point_norm(children_count)))  # 将颜色范围限制在0.3-0.7之间，使颜色更浅
            left, width = adjust_width_for_log_scale(row['bround'], width)
            # 计算对数尺度下的矩形边界
            l = 0.05 if type == MAXSAT else 1
            rect = Rectangle((left, min_ub-l), width, max_ub - min_ub+l*2, 
                            linewidth=1.5, edgecolor='#5494CE',  facecolor = color,#facecolor='#00b4d8',
                            linestyle='-', capstyle='round', joinstyle='round',
                            rotation_point='center',alpha = 0.4, zorder = 2)
            # rect = FancyBboxPatch((row['bround'] - 500, min_ub), 
            #                   1000, max_ub - min_ub, 
            #                   boxstyle="round,pad=0.2,rounding_size=100", linewidth=1, 
            #                   edgecolor='none', facecolor='#0096c7', linestyle='--')
            ax.add_patch(rect)
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
                ax.plot([pre_round, block_round, block_round], [pre_ub, pre_ub, block_ub], color = '#5494CE',
                        linestyle='--',linewidth = 0.5,alpha = 0.5, zorder = 0)#color='#CFCFCF'

    # # 处理Lowerbounds数据
    lb_rounds_new = list(map(int, lowerbounds.keys()))
    if type == MAXSAT:
        lb_values_new = [-v for v in lowerbounds.values()]
    elif type == TSP:
        lb_values_new = [v for v in lowerbounds.values()]
    elif type == MIPLTP:
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
        ax.scatter(point["bround"], pre_point["ub"], color=color, alpha=alpha, 
                    s =s ,zorder =  zorder, rasterized=rasterized,
                    edgecolor='none')
    
    def get_pre_point(point):
        pre_point_rows = ub_df[ub_df['pname'] == point['pre_pname']]
        if pre_point_rows.empty:
            print("not foundd pre", point['pname'])
            return point
        return pre_point_rows.iloc[0]       
    
    if type == TSP:
        for i, point in ub_df.iterrows():
            if point['pname'] not in sampled_points:
                continue  # 只绘制抽样的点
        #     # if point['block'] != "B1":
        #     #     continue
        #     # print(point['pname'])
        #     # if not point["allInteger"]:
        #     #     continue
        #     # if point['pname'] in sampled_points or (point["fathomed"] and not point["allInteger"]):
            pre_point = get_pre_point(point) if not point["allInteger"] else point
            draw_point(point, pre_point)

    # # 绘制Lowerbounds的线段
    ax.plot(lb_rounds_new, lb_values_new, color="#66A266", linewidth=8, zorder = 3)

    plts = [plt.Line2D([], [], color='green', linewidth=2, label='upper bounds'),
        plt.Line2D([],[],color="red", linestyle='None',marker = 'o', label="integer"),
        plt.Line2D([],[],color="#0072BD", linestyle='None',marker = 'o',  label="main-chain"),
        plt.Line2D([],[],color="orange", linestyle='None',marker = 'o',  label="fathomed"),
        plt.Line2D([],[],color="black", linestyle='None',marker = 'o', label="fork"),
        plt.Line2D([],[],color="#9acd32", linestyle='None',marker = 'o', label="unpublished")]
    
    ax.set_xlabel(None)
    ax.set_ylabel('Values')
    ax.set_xscale("log")
    
    fig.subplots_adjust(left=0.118, bottom=0.079, right=0.975, top=0.974)
    if type == TSP:
        ax.set_xlim(2000, 110000)
        ax.set_ylim(2700, 4000)
    elif type == MAXSAT:
        ax.set_xlim(100, 4500)
        ax.set_ylim(59, 63)
        # ax.set_xlim(1, 100)
        # ax.set_ylim(18, 21)
    elif type == MIPLTP:
        ax.set_xlim(10, 430)
        ax.set_ylim(180, 250)
    # plt.ylim(180, 250)
    # plt.xlim(-30, 430)
    # plt.grid(True)
    # plt.legend(handles = plts)
    ax.set_rasterized(True)
    plt.savefig(SAVE_PREFIX + f"\\bounds_{type}_m{m}_{time.strftime('%m%d_%H%M%S')}.svg", dpi=200)
    plt.show()

def create_fig3():
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    # 创建3x3的图
    # fig = plt.figure(figsize=(15, 15))
    # gs = GridSpec(3, 3, figure=fig)
    
    # 读取所有数据文件
    f1 = Path.cwd()/"Results/20250316/134735/pint24_conti24_ub24_eq10_gr4x6m1d1evaluation results.json"
    f2 = Path.cwd()/"Results/20250316/134727/pint24_conti24_ub24_eq10_gr4x6m3d1evaluation results.json"
    f3 = Path.cwd()/"Results/20250316/134140/pint24_conti24_ub24_eq10_gr4x6m5d1evaluation results.json"
    f4 = Path.cwd()/"Results/20250316/133345/pvar162_soft81_con162_pseudoBoolean-normalized-g9x9.opb.msatm1d1evaluation results.json"
    f5 = Path.cwd()/"Results/20250316/133803/pvar162_soft81_con162_pseudoBoolean-normalized-g9x9.opb.msatm3d1evaluation results.json"
    f6 = Path.cwd()/"Results/20250316/133936/pvar162_soft81_con162_pseudoBoolean-normalized-g9x9.opb.msatm5d1evaluation results.json"   
    f7 = Path.cwd()/"Result_Data/tspfig3/m1d5vtspevaluation results.json"
    f8 = Path.cwd()/"Result_Data/tspfig3/m3d5vtspevaluation results.json"
    f9 = Path.cwd()/"Result_Data/tspfig3/m5d5vtspevaluation results.json"

    
    # # 创建子图并绘制
    # axes = []
    # for i in range(3):
    #     for j in range(3):
    #         ax = fig.add_subplot(gs[i, j])
    #         axes.append(ax)
            
    # # MIPLTP问题
    # for idx, f in enumerate([f1, f2, f3]):
    #     plot_bounds_fig3(f, MIPLTP, axes[idx])
        
    # # MAXSAT问题
    # for idx, f in enumerate([f4, f5, f6]):
    #     plot_bounds_fig3(f, MAXSAT,axes[idx])
        
    # # TSP问题
    # for idx, f in enumerate([f7, f8, f9]):
    #     plot_bounds_fig3(f, TSP, axes[idx])
    
    # # 调整布局
    # plt.tight_layout()
    
    # # 保存图片
    # plt.savefig(SAVE_PREFIX + f"\\bounds_maxsat{time.strftime('%H%M%S')}.svg",  dpi=300)
    # plt.show()

if __name__ == "__main__":
    # create_fig3()
    f1 = Path.cwd()/"Results/20250316/134735/pint24_conti24_ub24_eq10_gr4x6m1d1evaluation results.json"
    f2 = Path.cwd()/"Results/20250316/134727/pint24_conti24_ub24_eq10_gr4x6m3d1evaluation results.json"
    f3 = Path.cwd()/"Results/20250316/134140/pint24_conti24_ub24_eq10_gr4x6m5d1evaluation results.json"
    f4 = Path.cwd()/"Results/20250316/133345/pvar162_soft81_con162_pseudoBoolean-normalized-g9x9.opb.msatm1d1evaluation results.json"
    f5 = Path.cwd()/"Results/20250316/133803/pvar162_soft81_con162_pseudoBoolean-normalized-g9x9.opb.msatm3d1evaluation results.json"
    f6 = Path.cwd()/"Results/20250316/133936/pvar162_soft81_con162_pseudoBoolean-normalized-g9x9.opb.msatm5d1evaluation results.json"   
    f7 = Path.cwd()/"Result_Data/tspfig3/m1d5vtspevaluation results.json"
    f8 = Path.cwd()/"Result_Data/tspfig3/m3d5vtspevaluation results.json"
    f9 = Path.cwd()/"Result_Data/tspfig3/m5d5vtspevaluation results.json"
    f10 = Path.cwd()/"Results/20250317/201726/pvar54_soft27_con54_pseudoBoolean-normalized-g9x3.opb.msatm1d1evaluation results.json"

    # f7 = Path.cwd()/ "Results\\20250319\\175110\\burma14m1d10evaluation results.json"
    # f8 = Path.cwd()/ "Results\\20250319\\175110\\burma14m3d10evaluation results.json"
    # f9 = Path.cwd()/ "Results\\20250319\\175110\\burma14m5d10evaluation results.json"
    f10 = Path.cwd()/ "Results\\20250319\\175110\\burma14m10d10evaluation results.json"


    # plot_bounds_fig3(f1, MIPLTP, m=1)
    # plot_bounds_fig3(f2, MIPLTP, m=3)
    # plot_bounds_fig3(f3, MIPLTP, m=5)
    # plot_bounds_fig3(f4, MAXSAT,m=1)
    # plot_bounds_fig3(f5, MAXSAT,m=3)
    # plot_bounds_fig3(f6, MAXSAT,m=5)
    # plot_bounds_fig3(f7, TSP,m=1)
    # plot_bounds_fig3(f8, TSP,m=3)
    # plot_bounds_fig3(f9, TSP,m=5)
    plot_bounds_fig3(f10, TSP,m=10)