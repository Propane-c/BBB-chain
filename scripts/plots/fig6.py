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
    plt.rcParams['font.size'] = 12
    df = pd.DataFrame([data for data in data_list if data["difficulty"] in [3,5,8,10]])
    df_sorted = df.sort_values(by=['safe_thre', 'difficulty'], ascending=[False, True])
    sorted_safe_thre = df_sorted['safe_thre'].unique()
    
    # 添加 safe_thre_value 列
    df['safe_thre_value'] = df['safe_thre']
    
    np.random.seed(0)  # 为了可重复性的示例
    sns.set(style="whitegrid", font='Times New Roman', font_scale=1)
    colors = ["#FF8283", "#0D898A", "#f9cc52", "#5494CE"]

    # 创建图表
    fig = plt.figure(figsize=(8, 9))
    grid = fig.add_gridspec(6, 2, height_ratios=[3, 1, 0.8, 2, 0.5, 1], width_ratios=[1, 1])
    
    # 创建子图
    ax1 = fig.add_subplot(grid[0, 0:2])
    ax3 = fig.add_subplot(grid[1, 0:2])
    axins1 = fig.add_subplot(grid[3, 0])
    axins2 = fig.add_subplot(grid[3, 1])
    ax2 = fig.add_subplot(grid[5, 0:2])
    
    # 添加子图标签
    ax1.text(-0.1, 1.02, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax3.text(-0.1, 1.02, 'b', transform=ax3.transAxes, fontsize=12, fontweight='bold')
    ax2.text(-0.1, 1.02, 'e', transform=ax2.transAxes, fontsize=12, fontweight='bold')
    axins1.text(-0.226, 1.02, 'c', transform=axins1.transAxes, fontsize=12, fontweight='bold')
    axins2.text(-0.17, 1.02, 'd', transform=axins2.transAxes, fontsize=12, fontweight='bold')

    ax1: plt.Axes
    ax41: plt.Axes
    ax42: plt.Axes 
    ax2: plt.Axes
    ax3: plt.Axes
    # ax3: plt.Axes
    
    # 选择特定的 difficulty，例如 difficulty 为 5
    specific_difficulty = 5
    df_specific = df[df['difficulty'] == specific_difficulty]

    # 第一个图表：Rate的柱状图
    
    original_xticks = [0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0003, 0.0001]
    # # 视觉上均匀分布的x轴刻度位置
    visual_xticks = range(0, len(original_xticks))
    ax1.set_xticks(visual_xticks)
    ax1.set_xticklabels([])
    ax1.plot(visual_xticks, 'safe_thre' , 
                data = df[df["difficulty"] == 5].sort_values(by="safe_thre",ascending=False),  
                marker='o', linestyle="--", color="red", label=None)
    
    # 添加文字说明
    last_x = visual_xticks[3]
    last_y = df[df["difficulty"] == 5].sort_values(by="safe_thre", ascending=False)['safe_thre'].iloc[3]
    ax1.text(last_x + 0.3, last_y, 'Secure Threshold', color='red', va='center')
    
    # 绘制柱状图
    bars = sns.barplot(x='safe_thre', y='ave_advrate', hue='difficulty', 
                palette=colors, data=df, ax=ax1, width=0.7, order=sorted_safe_thre)
    
    # 获取柱状图的图例句柄和标签
    handles, labels = ax1.get_legend_handles_labels()
    # 只保留柱状图的图例（移除红线的图例）
    ax1.legend(handles=handles[1:], labels=labels[1:], title='Difficulty', loc="upper right")
    ax1.set_xlabel(' ')
    ax1.set_ylabel('Success probability', labelpad=15)
    # ax1.legend(title='Difficulty', loc="upper right")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xticklabels([])
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.grid(axis='y')

    # 在ax1和ax3的y轴上添加方向箭头
    # ax1的向上箭头（在y轴的顶部）
    ax1.annotate('', xy=(0, 0.2), xytext=(0, 0),
                 xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # ax3的向下箭头（在y轴的底部）
    ax3.annotate('', xy=(0, 0.4), xytext=(0, 1),
                 xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=2))

    # 第二个图表：Prob的折线图
    # ax2.set_xticks(visual_xticks)
    # markers = ['o', 's', '^', 'D']  # 圆形、方形、三角形、菱形
    # for i, difficulty in enumerate(df['difficulty'].unique()):
    #     print(difficulty)
    #     ax2.plot(visual_xticks, 'ave_accept_advrate' , 
    #             data = df[df["difficulty"] == difficulty].sort_values(by="safe_thre",ascending=False),  
    #             marker=markers[i], linestyle="--", color=colors[i], label=f"Difficulty = {difficulty}" if i == 0 else difficulty)
    sns.barplot(x='safe_thre', y='ave_accept_advrate', hue='difficulty', 
                palette=colors, data=df, ax=ax2, width=0.7, order=sorted_safe_thre)
    ax2.set_ylabel('Chain\nquality', labelpad=8)
    ax2.set_xlabel('Secure threshold')  # 移除x轴标签，因为将与第二个图共享
    ax2.set_ylim(bottom=0 + 0.00001)
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    # ax2.legend(loc="upper right", ncol=4)
    ax2.get_legend().remove()
    ax2.grid(True)
    # ax2.set_xlim(ax1.get_xlim())
    # ax2.set_xticklabels(original_xticks)
    ax2.grid(axis='y')


    # 第三个图表：Security Margin的柱状图
    df['security_margin'] = df['safe_thre'] - df['ave_advrate']
    sns.barplot(x='safe_thre', y='security_margin', hue='difficulty', 
                palette=colors, data=df, ax=ax3, width=0.7, order=sorted_safe_thre)
    ax3.set_ylabel('Secure \nmargin', labelpad=8)
    ax3.set_xlabel('Secure Threshold')
    ax3.get_legend().remove()
    ax3.grid(True)
    ax3.set_yscale("log")
    ax3.grid(axis='y')
    ax3.invert_yaxis()

    df.loc[df['difficulty'] == 5, 'safe_ratio'] = \
        df.loc[df['difficulty'] == 5, 'ave_advrate'] / df.loc[df['difficulty'] == 5, 'safe_thre']
    df_d5 = df[df["difficulty"] == 5].sort_values(by="safe_thre", ascending=False)
    
    # original_xticks = [0.005, 0.003, 0.001, 0.0008, 0.0005, 0.0003, 0.0001]
    # # 视觉上均匀分布的x轴刻度位置
    # visual_xticks = range(0, len(original_xticks))
    # ax41.set_xticks(visual_xticks)
    # ax41.set_xticklabels([])
    # ax41.plot(visual_xticks, 'safe_thre' , 
    #             data = df[df["difficulty"] == 5].sort_values(by="safe_thre",ascending=False),  
    #             marker='x',linestyle = "--", color = "#0D898A",
    #             label = "threshlod")
    # ax41.plot(visual_xticks, 'ave_advrate' , 
    #             data = df[df["difficulty"] == 5].sort_values(by="safe_thre",ascending=False),  
    #             marker='o',color = "#0D898A",
    #             label = "simulation")
    # ax41.set_ylim(bottom=0, top=0.0051)
    # # ax41.set_xticks([])
    # # ax41.set_xticks(sorted_safe_thre)
    # ax41.set_xlim(ax1.get_xlim())
    # # print()
    # # print(ax3.get_xlim())
    # # ax41.invert_xaxis()
    # ax41.legend(loc = "upper left",bbox_to_anchor=(0.3, 1.01),ncol=2)
    # axins_2 = ax41.twinx()
    # axins_2.plot(visual_xticks, 'safe_ratio' , 
    #             data = df[df["difficulty"] == 5].sort_values(by="safe_thre",ascending=False), 
    #                 marker='o', color = "#BC5133",alpha = 0.5)
    # axins_2.spines['left'].set_color('#0D898A')  # Set the color of the y-axis to blue
    # # axins_2.set_xticks(sorted_safe_thre)
    # axins_2.set_xticks(visual_xticks)
    # # axins_2.set_xticks([])
    # axins_2.set_xticklabels([])
    # # ax_inset.set_yticks([0.00])
    # ax41.yaxis.label.set_color('#0D898A')
    # ax41.tick_params(axis='y', colors='#0D898A')
    # ax41.set_ylabel("Success\nprobability",labelpad=15)
    # ax41.set_xlabel(" ")
    # ax41.grid(axis='y')
    # # labels = [item.get_text() for item in ax41.get_xticklabels()]
    # ax41.tick_params(axis='x')
    # # ax41.xaxis.set_major_locator(plt.MaxNLocator(5))
    # axins_2.spines['right'].set_color('#BC5133')  # Set the color of the y-axis to blue
    # axins_2.yaxis.label.set_color('#BC5133')
    # axins_2.tick_params(axis='y', colors='#BC5133')
    # axins_2.set_ylabel("Safety performance")
    # axins_2.set_ylim(0.1, 0.4)
    # # ax_inset.grid(False)
    # axins_2.grid(False)

    # axins1 = ax1.inset_axes([0.33, 0.4, 0.3, 0.5])
    # axins2 = ax1.inset_axes([0.7, 0.4, 0.3, 0.5])
    # axins1 = fig.add_subplot(grid[3, 0])
    # axins2 = fig.add_subplot(grid[3, 1])

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
    axins1.set_ylabel("Success probability", labelpad = 15)
    axins1.tick_params(axis='x')
    axins1.tick_params(axis='y')
    axins1.set_xlabel("Blocks")
    axins2.set_ylabel(" ",labelpad = 15)
    axins2.set_xlabel("Blocks")
    axins2.tick_params(axis='y')
    axins2.tick_params(axis='x')
    axins1.legend()
    axins2.legend()
    fig.subplots_adjust(left=0.13, bottom=0.08, right=0.98, top=0.98,hspace=0.05)
    import time
    plt.savefig(f"E:\Files\A-blockchain\\branchbound\\figs\\secure{time.strftime('%Y%m%d%H%M%S')}.eps", dpi=300)
    # plt.show()
    
    # 在创建完所有子图后调整位置
    # pos_ax2 = ax2.get_position()
    # pos_axins1 = axins1.get_position()
    # pos_axins2 = axins2.get_position()
    
    # # 向上移动axins1和axins2
    # axins1.set_position([pos_axins1.x0, pos_axins1.y0 - 0.1, pos_axins1.width, pos_axins1.height])
    # axins2.set_position([pos_axins2.x0, pos_axins2.y0 - 0.1, pos_axins2.width, pos_axins2.height])

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


if __name__ == "__main__":
    plot_security_fig6()
