import json
import math
import pathlib
import sys
sys.path.append("E:\Files\gitspace\\bbb-github")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

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
    heights = [3, 1, 1]
    fig = plt.figure(figsize=(8, 6))
    # fig, (ax1, ax41, ax2, ax3) = plt.subplots(4, 1, figsize=(10, 8), 
    #                                     gridspec_kw={'height_ratios': heights, 'hspace': 0})
    
    grid = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], width_ratios=[1])
    ax1 = fig.add_subplot(grid[0, 0]) # ax_a will span two rows.
    ax41 = fig.add_subplot(grid[1, 0])
    # ax42 = fig.add_subplot(grid[1, 0])
    # ax43 = fig.add_subplot(grid[1, 2]) # Placeholder for ax_d (e in the description)
    ax2 = fig.add_subplot(grid[2, 0])
    # ax3 = fig.add_subplot(grid[3, 0])

    ax1: plt.Axes
    ax41: plt.Axes
    ax42: plt.Axes 
    ax2: plt.Axes
    # ax3: plt.Axes
    
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
    # ax2.set_xticklabels([])
    ax2.grid(True)
    ax2.grid(axis='y')
    ax2.invert_yaxis()
    # ax2.grid(True)

    # 调整子图间距并显示图表

    # 第三个图表：Wasted的箱型图
    
    # wasted_path = pathlib.Path(".\Result_Data\\1029attack_data2lite.json")
    # with open(wasted_path,'r') as f:
    #     data_list = []
    #     jsondata_list = f.read().split('\n')[:-1]
    #     for jsondata in jsondata_list:
    #         data_list.append(json.loads(jsondata))
    #     wasted_df = pd.DataFrame(data_list)
    #     print(wasted_df[wasted_df["difficulty"] == 5])
    # sns.boxplot(x='safe_thre', y="ave_subpair_unpubs", data = wasted_df[wasted_df["difficulty"] == 5], 
    #             ax=ax3, width = 0.2, linewidth = 2,order=sorted_safe_thre)
    # ax3.set_ylabel('Wasted\nworkload',labelpad = 22)
    ax2.set_xlabel('Safe threshold')
    # ax3.grid(True)
    # ax3.grid(axis='y')
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
    axins_2.set_ylim(0.1, 0.4)
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