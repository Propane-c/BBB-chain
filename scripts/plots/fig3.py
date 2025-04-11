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
FANGDA  = "fangda"

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
            fig = plt.figure(figsize=(12, 6))
        elif type == FANGDA:
            fig = plt.figure(figsize=(6, 5))
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

    def draw_int_path(path, color = "#FF8283", linewidth=1.8, linestyle='--'):
        for i in range(len(path) - 1):
            start_point = (path[i]['bround'], path[i]['ub'])
            end_point = (path[i + 1]['bround'], path[i + 1]['ub'])
            ax.plot([start_point[0], start_point[0], end_point[0]], [start_point[1], end_point[1], end_point[1]], 
                    color = color, linestyle=linestyle, linewidth=linewidth, alpha=0.9, zorder=6)


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
    solve_rounds = data['solve_rounds']
    max_round = solve_rounds["B0"] if solve_rounds else 0
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
    
    print("loading data")
    # 创建数据帧
    ub_df = pd.DataFrame(ub_data)
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
    bpre_df = pd.concat(bpre_rows)
    children_counts[((0,0),0)]=bpre_df['children_count'].max()

    # sns.set(style="white")
    pink_to_red_colors = [
        # "#FFC0CB",  
        # "#FFB6C1",  
        # "#FF9999",  
        "#FF7F7F",  
        "#FF6666",  
        "#FF4D4D",  
        "#FF3333",  
        # "#FF1A1A",  
        "#FF0000",
        "#D62728" 
    ]
    print("drawing integer path")
    # 对于相同的整数解，只保留最早出现的点
    int_df = ub_df[(ub_df["allInteger"] == True) & (ub_df["block"]!= "None")].copy()
    int_df = int_df.sort_values('bround')  # 按轮次排序
    int_df = int_df.drop_duplicates(subset=['ub'], keep='first')  # 对于相同的ub值只保留第一次出现的点
    # 标记allInteger路径
    int_paths = []
    intpath_points = set()
    max_bround_point = None
    max_bround = float('-inf')
    # 按bround值排序，确定每个点的颜色索引
    sorted_brounds = sorted(int_df['bround'].unique())
    bround_to_color_index = {
        b: int((i / (len(sorted_brounds) - 1 if len(sorted_brounds) > 1 else 1)) * (len(pink_to_red_colors) - 1))
        for i, b in enumerate(sorted_brounds)
    }
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
            start_point = path[0]  # path[0]是最新的点(bround最大)
            bround = start_point['bround']
            
            color_index = bround_to_color_index[bround]
            color_index = min(color_index, len(pink_to_red_colors) - 1)
            path_color = pink_to_red_colors[color_index]
            
            if type == FANGDA:
                draw_int_path(path, path_color, 10, '--')
            else:
                draw_int_path(path, path_color, 4, '--')
        else:
            start_point = path[0]
            bround = start_point['bround']
            color_index = bround_to_color_index[bround]
            color_index = min(color_index, len(pink_to_red_colors) - 1)
            path_color = pink_to_red_colors[color_index]
            
            # 线宽稍小一些，区分主要路径
            if type == FANGDA:
                draw_int_path(path, path_color, 8, '--')
            else:
                draw_int_path(path, path_color, 2, '--')
    
    
    

    print("drawing points")
    sampled_points = set(root_df['pname']).union(intpath_points).union((((0,0),0)))
    # sampled_points = set(ub_df.sample(frac=0.1, random_state=0)['pname'])
    # sampled_points = sampled_points.union(intpath_points)
    # ["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"] 
    # ["#FF8283", "#0D898A","#f9cc52","#5494CE", ] '#00796B' '#ff8c00' '#b22222'
    miner_colors = [
        "#00B0F0", 
        "#00FA9A", 
        "#FFF700", 
        "#00CED1", 
        "#32CD32", 
        "#FFD700", 
        "#0096FF", 
        "#0D898A", 
        "#FFBF00", 
        "#4169E1", 
    ]
    # 绘制UB数据点和连接线
    smain = 15
    smain = 15 if type == MAXSAT else 40
    if type == TSP:
        s = 15
        sfork = 40
        sorange = 3
        sopt = 100
    elif type == MAXSAT:
        s = 15
        sfork = s
        sorange = 15
        sopt = 80
    elif type == MIPLTP:
        s = 40
        sfork = s
        sorange = 15
        sopt = 60
    elif type == FANGDA:
        s = 100
        sfork = 200
        sorange = 15
        sopt = 400
    rasterized=False if type == MIPLTP else False 

    if type != TSP and type != FANGDA:
        # sns.scatterplot(x="bround",y ="ub",
        #                 data = ub_df[(ub_df["fathomed"] == False) & (ub_df["block"]!= "None")] ,
        #                 s = smain, color = '#00B0F0', rasterized=rasterized ,edgecolor="none",
        #                 zorder = 5, alpha = 0.7)

        # 按矿工ID分组绘制主链数据
        for miner_id in ub_df['miner'].unique():
            miner_data = ub_df[
                (ub_df['miner'] == miner_id) & 
                (ub_df['fathomed'] == False) & 
                (ub_df['block'] != 'None')
            ]
            if not miner_data.empty:
                sns.scatterplot(
                    x="bround", y="ub",
                    data=miner_data,
                    s=smain, 
                    color=miner_colors[miner_id], 
                    rasterized=rasterized,
                    edgecolor="none",
                    zorder=5, 
                    alpha=0.7
                )

    sns.scatterplot(x="bround",y ="ub",
                    data = ub_df[(ub_df["fathomed"]== True) & 
                                (ub_df["allInteger"]==False) & (ub_df["block"]!= "None")] , 
                    color = "#CFCFCF", s= sorange, rasterized=rasterized, edgecolor="none",alpha = 0.5)
    # sns.scatterplot(x="bround",y ="ub",
    #                 data = ub_df[(ub_df["fathomed"]== True) & 
    #                              (ub_df["allInteger"]==False) & (ub_df["block"]!= "None")] , #ff8c00#FF9E4A#EDB120#ECE3AFFF#FFD700#DA70D6
    #                 color = "", s= s, rasterized=rasterized, edgecolor="none",alpha = 0.5)#E6D5BE #ee9b00#9acd32 #E2CAA9FF
    unpub_zorder = 4 if type == FANGDA or type == TSP else 2
    # sns.scatterplot(x="round",y ="ub",data = ub_df[(ub_df["block"]== "None")], 
    #                 color = "#39FF14", s= s,rasterized=rasterized,edgecolor="none",zorder = unpub_zorder, alpha = 1)
    sns.scatterplot(x="bround",y ="ub",data = ub_df[(ub_df["isFork"]== True) & (ub_df["block"]!= "None")] , 
                    color = "#FF69B4", s= sfork,rasterized=rasterized,edgecolor="none",zorder = 4, alpha = 1)#FF9E4A
    bright_colors = {
        "青色": "#00FFFF",      
        "霓虹绿": "#39FF14",    
        "亮粉色": "#FF69B4",     
        "柠檬黄": "#FFF700",     
        "青柠色": "#BFFF00",     
        "珊瑚红": "#FF7F50",     
        "霓虹蓝": "#1E90FF",     
        "亮绿松": "#00FA9A",     
        "霓虹紫": "#9D00FF",     
        "青绿色": "#00CED1"      
    }

    for _, point in int_df.iterrows():
        bround = point['bround']
        color_index = bround_to_color_index[bround]
        color_index = min(color_index, len(pink_to_red_colors) - 1)
        scatter_color = pink_to_red_colors[color_index]
        print(bround, point['ub'])
        ax.scatter(
            bround, 
            point['ub'], 
            s=sopt, 
            color=scatter_color, 
            edgecolor="none", 
            zorder=6,
            rasterized=rasterized
        )
    


    print("drawing main chain")
    base_colors = {
        'blue': [
            "#00B0F0",
            "#0096FF",
            "#1E90FF",
            "#4169E1",
            "#0047AB" 
        ],
        'green': [
            "#00FA9A",  
            "#3CB371",  
            "#32CD32",  
            "#228B22",  
            "#008B45"   
        ],
        'yellow': [
            "#FFF700", 
            "#FFD700", 
            "#FFBF00", 
            "#FFA500", 
            "#FF8C00"  
        ]
    }

    def get_miner_color(miner_id: int, total_miners: int):
        """根据矿工ID获取颜色，同一基色使用不同色调"""
        base_idx = miner_id % 3
        color_group = miner_id // 3
        
        if base_idx == 0:
            colors = base_colors['blue']
        elif base_idx == 1:
            colors = base_colors['green']
        else:
            colors = base_colors['yellow']
        
        return colors[color_group % len(colors)]

    point_norm = mcolors.Normalize(vmin=0, vmax=bpre_df['children_count'].max())
    # print(bpre_df['children_count'].max())
    blues = plt.cm.Blues
    drawRect = True if type != TSP and type != FANGDA else False
    point_norm = mcolors.Normalize(vmin=0, vmax=math.log(max(children_counts.values())))
    my_blues = mcolors.LinearSegmentedColormap.from_list("my_blues", 
    ["#caf0f8","#caf0f8","#ade8f4","#90e0ef","#48cae4","#00b4d8","#00B0F0", "#0096c7","#0077b6"]) # "", ,"#003049"])#"#caf0f8" ,#00B0F0
    def adjust_width_for_log_scale(center_x, desired_width, base=10):
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
            # print(i)
            # print(row)
        # 提取每个block的数据
            blocks.append(row['block'])
            min_ub = row['ub_min']
            max_ub = row['ub_max']
            # 绘制圆角矩形
            if type == MIPLTP:
                width = 0.04
                if m == 1:
                    width = 0.04
                elif m == 3:
                    width = 0.035
                elif m == 10:
                    width = 0.03
            elif type == MAXSAT:
                width = 0.023
                if m == 1:
                    width = 0.023
                elif m == 3:
                    width = 0.02
                elif m == 10:
                    width = 0.015
            # children_count = children_counts[row['pre_pname']]
            color = miner_colors[row['miner']]
            # color = my_blues(0 + 0.5*(1-point_norm(children_count)))
            left, width = adjust_width_for_log_scale(row['bround'], width)
            # left = row['bround']-width/2
            # 计算对数尺度下的矩形边界
            l = 0.05 if type == MAXSAT else 1
            rect = Rectangle((left, min_ub-l), width, max_ub - min_ub+l*2, 
                            linewidth=1.5, edgecolor='#5494CE',  facecolor = color,
                            linestyle='-', capstyle='round', joinstyle='round',
                            rotation_point='center',alpha = 0.2, zorder = 2)
            ax.add_patch(rect)
        if type == MIPLTP:
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
    lb_rounds_new = [x-1 if x != 0 else x for x in map(int, lowerbounds.keys())]
    if type == MAXSAT:
        lb_values_new = [-v for v in lowerbounds.values()]
    elif type == TSP or type == FANGDA:
        lb_values_new = [v for v in lowerbounds.values()]
    elif type == MIPLTP:
        lb_values_new = [v for v in lowerbounds.values()]
    # my_oranges = mcolors.LinearSegmentedColormap.from_list("my_oranges", ["white", "#ee9b00"])
    
    def draw_point(point,pre_point):
        alpha=0.2 if m == 1 else 0.4
        color = '#0072BD'  # 默认颜色
        if math.log(point['children_count']) > 10:
            zorder = 1
        elif math.ceil(math.log(point['children_count'])) > 5:
            zorder = 3
        elif math.ceil(math.log(point['children_count'])) > 3:
            zorder = 2
        elif math.ceil(math.log(point['children_count'])) > 1:
            zorder = 2
        elif math.ceil(math.log(point['children_count'])) >= 0:
            zorder = 2  
            alpha = 0.4 if m == 1 else 0.6
        if type == TSP:
            s = ((point['children_count'])**0.5)*10+5
        elif type == FANGDA:
            s = ((point['children_count'])**0.5)*30+10
        # color = my_blues(0.1 + 0.75 *(1- point_norm(math.log(point['children_count']))))

        miner_id = point['miner']
        color = miner_colors[miner_id]  # 使用基础颜色循环
        
        if (pre_point["allInteger"] or (pre_point["fathomed"] and not point["allInteger"]) 
            or pre_point["isFork"] or pre_point["block"] == "None"):
            return
        ax.scatter(point["bround"], pre_point["ub"], color=color, alpha=alpha, 
                    s =s ,zorder =  zorder, rasterized=rasterized,
                    edgecolor='none')
    
    def get_pre_point(point):
        pre_point_rows = ub_df[ub_df['pname'] == point['pre_pname']]
        if pre_point_rows.empty:
            # print("not foundd pre", point['pname'])
            return point
        return pre_point_rows.iloc[0]       
    
    if type == TSP or type == FANGDA:
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
    bound_line_width = 6 if type != FANGDA else 10
    ax.plot(lb_rounds_new, lb_values_new, color="#333333", linewidth=bound_line_width, zorder = 5)
    # ax.plot(lb_rounds_new, lb_values_new, color="#60636A", linewidth=5, zorder = 3)#66A266

    plts = [plt.Line2D([], [], color='green', linewidth=2, label='upper bounds'),
        plt.Line2D([],[],color="red", linestyle='None',marker = 'o', label="integer"),
        plt.Line2D([],[],color="#0072BD", linestyle='None',marker = 'o',  label="main-chain"),
        plt.Line2D([],[],color="orange", linestyle='None',marker = 'o',  label="fathomed"),
        plt.Line2D([],[],color="black", linestyle='None',marker = 'o', label="fork"),
        plt.Line2D([],[],color="#9acd32", linestyle='None',marker = 'o', label="unpublished")]
    
    ax.set_xlabel(None)
    ax.set_ylabel('Values')
    if not type == FANGDA:
        ax.set_xscale("log")
    
    if type == FANGDA:
        fig.subplots_adjust(left=0.166, bottom=0.102, right=0.936, top=0.974)
    elif type == TSP:
        fig.subplots_adjust(left=0.09, bottom=0.102, right=0.975, top=0.974)
    elif type == MIPLTP:
        fig.subplots_adjust(left=0.112, bottom=0.102, right=0.975, top=0.974)
    elif type == MAXSAT:
        fig.subplots_adjust(left=0.09, bottom=0.102, right=0.96, top=0.974)
    if type == TSP:
        # ax.set_xlim(96, 110000)
        ax.set_xlim(96, max_round+1000//m)
        # ax.set_xlim(2500, 110000)
        ax.set_ylim(2700, 4000)
    elif type == FANGDA:
        if m == 1:
            ax.set_xlim(50000, 56000)
            ax.set_ylim(3000, 3600)
        elif m == 3:
            ax.set_xlim(28000, 31500) # 29649
            ax.set_ylim(3000, 3600)
        elif m == 5:
            ax.set_xlim(1500, 3000) # 
            ax.set_ylim(3000, 3600)
        elif m == 10:
            ax.set_xlim(1600,2500)
            ax.set_ylim(3000, 3600)# 2132
    elif type == MAXSAT:
        # ax.set_xlim(100, 4500)
        ax.set_xlim(60, max_round+10)
        ax.set_ylim(59, 63)
        # ax.set_xlim(1, 100)
        # ax.set_ylim(18, 21)
    elif type == MIPLTP:
        # ax.set_xlim(10, 430)
        ax.set_xlim(10, max_round+5)
        ax.set_ylim(180, 250)
    # plt.ylim(180, 250)
    # plt.xlim(-30, 430)
    # plt.grid(True)
    # plt.legend(handles = plts)
    ax.set_rasterized(False)
    print("end")
    plt.savefig(SAVE_PREFIX + f"\\bounds_{type}_m{m}_{time.strftime('%m%d_%H%M%S')}.svg", dpi=300)
    # plt.show()

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

def create_legend(type=None):
    """创建MIPLIB和TSP的图例"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.set_axis_off()
    miner_colors = [
        "#00B0F0", 
        "#00FA9A", 
        "#FFF700", 
        "#00CED1", 
        "#32CD32", 
        "#FFD700", 
        "#0096FF", 
        "#0D8A32FF", 
        "#FDBE01FF", 
        "#4169E1", 
    ]
    
    if type == "MIPLIB":
        # 创建所有图例元素
        legend_elements = [
            plt.scatter([], [], c=miner_colors[i], s=40, 
                       label=f'Solver {i+1}', alpha=0.7)
            for i in range(10)
        ]
        # 添加其他元素
        legend_elements.extend([
            plt.scatter([], [], c='#CFCFCF', s=15, label='Fathomed', alpha=0.5),
            plt.scatter([], [], c='#FF69B4', s=40, label='Fork', alpha=1.0),
            plt.scatter([], [], c='#D62728', s=40, label='Integer', alpha=1.0),
            # plt.Line2D([], [], color='#D62728', linestyle='--', linewidth=2, label='Integer path'),
            plt.Line2D([], [], color='#333333', linewidth=2, label='Bounds'),
            # plt.Rectangle((0,0), 1, 1, facecolor=miner_colors[0], edgecolor='#5494CE', alpha=0.2, linewidth=1, label='Block')
        ])
        
        # 重新排列元素顺序
        ncols = 5
        nrows = (len(legend_elements) + ncols - 1) // ncols
        new_elements = []
        for col in range(ncols):
            for row in range(nrows):
                idx = row * ncols + col
                if idx < len(legend_elements):
                    new_elements.append(legend_elements[idx])
        legend_elements = new_elements
        
    elif type == "TSP":
        legend_elements = [
            plt.scatter([], [], c=miner_colors[i], s=40, 
                       label=f'Solver {i+1}', alpha=0.7)
            for i in range(10)
        ]
        # 添加其他元素
        legend_elements.extend([
            plt.scatter([], [], c='#CFCFCF', s=15, label='Fathomed', alpha=0.5),
            plt.scatter([], [], c='#FF69B4', s=40, label='Fork', alpha=1.0),
            plt.scatter([], [], c='#D62728', s=40, label='Integer', alpha=1.0),
            # plt.Line2D([], [], color='#D62728', linestyle='--', linewidth=2, label='Integer path'),
            plt.Line2D([], [], color='#333333', linewidth=2, label='Bounds')
        ])
        
        # 重新排列元素顺序
        ncols = 5
        nrows = (len(legend_elements) + ncols - 1) // ncols
        new_elements = []
        for col in range(ncols):
            for row in range(nrows):
                idx = row * ncols + col
                if idx < len(legend_elements):
                    new_elements.append(legend_elements[idx])
        legend_elements = new_elements

    ax.legend(handles=legend_elements, 
             loc='center', 
             ncol=ncols,
             edgecolor='none', 
             fancybox=True,
             frameon=False,
            # handletextpad=0.3,
            # columnspacing=0.8,
             bbox_to_anchor=(0.5, 0.5))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # create_fig3()
    f1 = Path.cwd()/"Results/20250316/pint24_conti24_ub24_eq10_gr4x6m1d1evaluation results.json"
    f2 = Path.cwd()/"Results/20250316/pint24_conti24_ub24_eq10_gr4x6m3d1evaluation results.json"
    f3 = Path.cwd()/"Results/20250316/pint24_conti24_ub24_eq10_gr4x6m10d1evaluation results.json"
    f4 = Path.cwd()/"Results/20250316/pvar162_soft81_con162_pseudoBoolean-normalized-g9x9.opb.msatm1d1evaluation results.json"
    f5 = Path.cwd()/"Results/20250316/pvar162_soft81_con162_pseudoBoolean-normalized-g9x9.opb.msatm3d1evaluation results.json"
    f6 = Path.cwd()/"Results/20250316/pvar162_soft81_con162_pseudoBoolean-normalized-g9x9.opb.msatm10d1evaluation results.json"   

    f11 = Path.cwd()/ "Results\\20250319\\175110\\burma14m1d10evaluation results.json"
    f12 = Path.cwd()/ "Results\\20250319\\175110\\burma14m3d10evaluation results.json"
    f13 = Path.cwd()/ "Results\\20250319\\175110\\burma14m5d10evaluation results.json"
    f14 = Path.cwd()/ "Results\\20250319\\175110\\burma14m10d10evaluation results.json"


    f7 = Path.cwd()/ "Results\\20250319\\184042\\burma14m1d5evaluation results.json"
    f8 = Path.cwd()/ "Results\\20250319\\184042\\burma14m3d5evaluation results.json"
    f9 = Path.cwd()/ "Results\\20250319\\184042\\burma14m5d5evaluation results.json"
    f10 = Path.cwd()/ "Results\\20250319\\184042\\burma14m10d5evaluation results.json"

    # plot_bounds_fig3(f1, MIPLTP, m=1)
    # plot_bounds_fig3(f2, MIPLTP, m=3)
    # plot_bounds_fig3(f3, MIPLTP, m=10)
    plot_bounds_fig3(f4, MAXSAT,m=1)
    plot_bounds_fig3(f5, MAXSAT,m=3)
    plot_bounds_fig3(f6, MAXSAT,m=10)
    # plot_bounds_fig3(f7, TSP,m=1)
    # plot_bounds_fig3(f8, TSP,m=3)
    # plot_bounds_fig3(f10, TSP,m=10)
    # plot_bounds_fig3(f11, TSP,m=1)
    # plot_bounds_fig3(f12, TSP,m=3)
    # plot_bounds_fig3(f13, TSP,m=5)
    # plot_bounds_fig3(f14, TSP,m=10)
    # plot_bounds_fig3(f7, FANGDA,m=1)
    # plot_bounds_fig3(f8, FANGDA,m=3)
    # plot_bounds_fig3(f10, FANGDA,m=10)
    # create_legend("MIPLIB")  # 创建MIPLIB的图例
    # create_legend("TSP")     # 创建TSP的图例