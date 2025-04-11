import json
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import pathlib

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


def create_data_list_searchst(json_data_list):
    """对应长链仿真，keyblock的分叉率"""
    data_dict = defaultdict()
    data_list = []
    for entry in json_data_list:
        if entry['openblk_st'] == OB_RAND and entry['openprblm_st'] == OP_BEST:
            strategy = 'BFS'
        if entry['openblk_st'] == OB_DEEP and entry['openprblm_st'] == OP_RAND:
            strategy = 'DFS'
        if entry['openblk_st'] == OB_BREATH and entry['openprblm_st'] == OP_RAND:
            strategy = 'BrFS'
        if entry['openblk_st'] == OB_RAND and entry['openprblm_st'] == OP_RAND:
            strategy = 'Rand'
        var_num = entry['var_num']
        difficulty = entry['difficulty']
        miner_num = entry['miner_num']
        # kb_strategy = entry['kb_strategy']
        total_mb_forkrate = entry['total_mb_forkrate']
        ave_solve_round = entry["ave_solve_round"]
        ave_subpair_num = entry["ave_subpair_num"]
        ave_subpair_unpubs = entry["ave_subpair_unpubs"]
        ave_acp_subpair_num = entry["ave_acp_subpair_num"]
        # mb_times = entry['mb_times']
        # data_dict[(strategy,miner_num)] = {
        #     "total_mb_forkrate": total_mb_forkrate,
        #     "ave_solve_round": ave_solve_round,
        #     "ave_subpair_num": ave_subpair_num,
        #     "ave_subpair_unpubs":ave_subpair_unpubs,
        #     "ave_acp_subpair_num":ave_acp_subpair_num,}
        data_list.append({
            "miner_num":miner_num,
            "strategy":strategy,
            "Mini-block\nForkrate": total_mb_forkrate,
            "Average Solving Rounds": ave_solve_round,
            "Workload on Chain": ave_subpair_num,
            "Workload\nwith Wasted":ave_subpair_unpubs,
            "Effective Workload":ave_acp_subpair_num,})
    return data_list


def draw_radar_chart(grouped_data:pd.DataFrame, miner_num, ax:plt.Axes):
        entries = grouped_data.columns.tolist()
        N = len(entries)

        angles = [(n / float(N) * 2 * math.pi + math.pi/2) % (2 * math.pi) for n in range(N)]
        angles += angles[:1]
        # fig, ax = plt.subplots(figsize=(10, 6.5), subplot_kw=dict(polar=True))
        
        ax.set_xticks(angles[:-1], entries)
        # 调整0度和180度的标签位置
        # 隐藏特定角度的原有标签并使用text函数添加新标签
        for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
            if angle!=0.5 * math.pi:
                label.set_visible(False)
                if angle == 0 or angle == 2 * math.pi:
                    y_offset = 0.28
                else:  # angle == math.pi
                    y_offset = 0.28
                ax.text(angle, 1+y_offset, label.get_text(), 
                        horizontalalignment='center',
                        verticalalignment="center", 
                        transform=ax.get_xaxis_transform())

        ax.set_rlabel_position(180)
        ax.set_yticks([], [], color="grey", size=7)
        ax.set_ylim(0, 1)

        colors = ['b', 'g',  'c','r', 'm', 'y', 'k']
        color_index = 0

        for strategy, st_data in grouped_data.iterrows():
            stats = st_data.to_list()
            stats += stats[:1]
            current_color = colors[color_index % len(colors)]
            ax.plot(angles, stats, color=current_color, linewidth=1.5, linestyle='solid', label=strategy)
            ax.fill(angles, stats, color=current_color, alpha=0.1)
            color_index += 1

        ax.legend(loc='upper left', bbox_to_anchor=(0.95, 0.5))# 减小色块宽度5,handlelength=1.0 


def draw_radars():
    """对应长链仿真，keyblock的分叉率"""
    file_path = pathlib.Path.cwd() / "Result_Data\\1210short_data测试不同st.json"
    
    jsondata_list = []
    with open(file_path, 'r') as f:
        jsons = f.read().split('\n')[:-1]
        for jsondata in jsons:
            jsondata_list.append(json.loads(jsondata))

    data_list = []
    for entry in jsondata_list:
        if entry['openblk_st'] == OB_RAND and entry['openprblm_st'] == OP_BEST:
            strategy = 'BFS'
        if entry['openblk_st'] == OB_DEEP and entry['openprblm_st'] == OP_RAND:
            strategy = 'DFS'
        if entry['openblk_st'] == OB_BREATH and entry['openprblm_st'] == OP_RAND:
            strategy = 'BrFS'
        if entry['openblk_st'] == OB_RAND and entry['openprblm_st'] == OP_RAND:
            strategy = 'Rand'
        miner_num = entry['miner_num']
        # kb_strategy = entry['kb_strategy']
        total_mb_forkrate = entry['total_mb_forkrate']
        ave_solve_round = entry["ave_solve_round"]
        ave_subpair_num = entry["ave_subpair_num"]
        ave_subpair_unpubs = entry["ave_subpair_unpubs"]
        ave_acp_subpair_num = entry["ave_acp_subpair_num"]
        data_list.append({
            "miner_num":miner_num,
            "strategy":strategy,
            "Fork\nPerformance": 1-total_mb_forkrate,
            "Solve Speed": 1/ave_solve_round,
            "Workload on Chain": ave_subpair_num,
            "Efficincy": ave_acp_subpair_num/ave_subpair_unpubs,
            "Total\nWorkload\n(Inverse)":1/ave_subpair_unpubs,
            "Effective Workload\n(Inverse)":1/ave_acp_subpair_num,})
         
    df = pd.DataFrame(data_list)
    grouped = df.groupby(['miner_num', 'strategy']).mean()[
        ['Solve Speed', 'Fork\nPerformance',  "Efficincy",
         'Effective Workload\n(Inverse)', 'Total\nWorkload\n(Inverse)',]]
    # 选择要归一化的列
    grouped_max = df.groupby('miner_num')[[
        'Solve Speed', 'Fork\nPerformance',  "Efficincy",'Effective Workload\n(Inverse)', 'Total\nWorkload\n(Inverse)']].max()
    normalized_grouped = grouped.copy()
    for column in grouped.columns:
        if column in grouped_max.columns:
            for miner_num in grouped_max.index:
                max_value = grouped_max.loc[miner_num, column]
                normalized_grouped.loc[(normalized_grouped.index.get_level_values('miner_num') == miner_num), column] /= max_value
        else:
            normalized_grouped[column] = grouped[column]
    
    # 设置字体为 Times New Roman，字号为12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12  # 调整字号大小
    # 创建子图并添加标签
    fig = plt.figure(figsize=(10, 8.5))
    gs = fig.add_gridspec(2, 2)
    
    # 定义标签位置
    label_x = -0.12  # 标签的x位置
    label_y = 1.01   # 标签的y位置
    
    # 创建所有子图
    ax1 = fig.add_subplot(gs[0, 0], polar=True) 
    ax2 = fig.add_subplot(gs[0, 1], polar=True)            
    ax3 = fig.add_subplot(gs[1, 0], polar=True)        
    ax4 = fig.add_subplot(gs[1, 1], polar=True)   


    # 添加标签
    ax1.text(label_x, label_y, 'a', transform=ax1.transAxes, 
                       fontsize=14, fontweight='bold')
    ax2.text(label_x, label_y, 'b', transform=ax2.transAxes, 
                  fontsize=14, fontweight='bold')
    ax3.text(label_x, label_y, 'c', transform=ax3.transAxes, 
                   fontsize=14, fontweight='bold')
    ax4.text(label_x, label_y, 'd', transform=ax4.transAxes, 
                        fontsize=14, fontweight='bold')
    
    miner_nums = [1, 3, 10, 30]
    for miner_num, ax in zip(miner_nums, [ax1, ax2, ax3, ax4]):
        if miner_num in normalized_grouped.index.get_level_values('miner_num'):
            sts_data = normalized_grouped.xs(miner_num, level='miner_num')
            draw_radar_chart(sts_data, miner_num, ax) 

    # plt.savefig(pathlib.Path.cwd() / "Result_Data\\1210short_data测试不同st.png", dpi=300)
    fig.subplots_adjust(left=0.09, bottom=0, right=0.88, top=1, hspace=0, wspace=0.76)
    plt.show()
    plt.close()

if __name__ == "__main__":
    draw_radars()