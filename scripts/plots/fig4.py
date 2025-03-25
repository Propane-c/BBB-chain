import json
import pathlib
import sys
sys.path.append("E:\Files\gitspace\\bbb-github")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import matplotlib.path as mpath
import matplotlib.patches as mpatches

SAVE_PREFIX = "E:\Files\A-blockchain\\branchbound\\branchbound仿真\\0129"
pathlib.Path.mkdir(pathlib.Path(SAVE_PREFIX), exist_ok=True)
SAVE = True

def plot_solveround_workload_fig4(file_path: str = None):
    """
    绘制求解轮数和工作量相关的图表
    
    Args:
        file_path: 数据文件路径，默认为 "Result_Data/1226v100_50m1_20.json"
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times new roman']
    plt.rcParams['font.size'] = 12
    colors = ["#f9cc52", "#5494CE","#FF8283", "#0D898A"]
    # sns.set(style="whitegrid")
    def plot_solve_rounds(ax:plt.Axes, df:pd.DataFrame, base_color:str, label:str, marker:str='o'):
        # 保留原有的色块代码但注释掉
        """
        # 原有的色块代码
        df_expanded = df.explode('solve_rounds')
        stats = df_expanded.groupby('miner_num')['solve_rounds'].agg(
            ['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]).reset_index()
        stats.columns = ['miner_num', 'median', 'q1', 'q3']
        n_points = len(stats)
        color_list = [mcolors.to_rgba(base_color, alpha) 
                     for alpha in np.linspace(1, 0.3, n_points)]
        
        for i in range(len(stats)-1):
            x_fill = [stats['miner_num'][i], stats['miner_num'][i+1]]
            y1_fill = [stats['q1'][i], stats['q1'][i+1]]  # 下四分位数
            y2_fill = [stats['q3'][i], stats['q3'][i+1]]  # 上四分位数
            ax.fill_between(x_fill, y1_fill, y2_fill, 
                          color=color_list[i], alpha=0.2)
            ax.plot(stats['miner_num'][i:i+2], stats['median'][i:i+2], 
                   color=color_list[i], linewidth=1.5)
            ax.plot(stats['miner_num'][i], stats['median'][i], 
                   marker=marker, color=color_list[i])
        """

        # 新增箱琴图代码
        df_expanded = df.explode('solve_rounds')
        positions = sorted(df['miner_num'].unique())
        
        # 创建箱琴图数据
        data = [df_expanded[df_expanded['miner_num'] == x]['solve_rounds'] for x in positions]
        
        # 绘制箱琴图
        parts = ax.violinplot(data, positions=positions, 
                            showmeans=False, showmedians=True)
        
        # 设置箱琴图样式
        for pc in parts['bodies']:
            pc.set_facecolor(base_color)
            pc.set_alpha(0.3)
        
        parts['cmedians'].set_color(base_color)
        
        # 添加图例项
        ax.plot([], [], color=base_color, label=label, marker=marker)

        # 设置坐标轴等
        ax.set_xlabel('Number of solvers')
        ax.set_ylabel('Solving rounds')
        ax.grid(True, linestyle='--', alpha=0.7)

    def plot_solve_rounds_with_stats(ax:plt.Axes, df_easy:pd.DataFrame, df_med:pd.DataFrame, df_hard:pd.DataFrame):
        """绘制求解轮数的统计图，包含中位数和四分位数"""
        # 读取数据
        # # file_path = pathlib.Path.cwd() / "Result_Data/1226v100_50m1_20_full.json"
        # file_path = pathlib.Path.cwd() / "E:\Files\gitspace\\bbb-github\Results\\20250306\\203200\\med_results.json"
        # data_list = []
        # with open(file_path, 'r') as f:
        #     jsondata_list = f.read().split('\n')[:-1]
        #     for jsondata in jsondata_list:
        #         data_list.append(json.loads(jsondata))
        # df = pd.DataFrame(data_list)
        # df = df.explode('solve_rounds')
        # df['solve_rounds'] = df['solve_rounds'].astype(float)
        # df = df.sort_values(by=["var_num", "difficulty", "miner_num"])
        # df_med = df[(df['difficulty'] == 5) & (df['var_num'] == 150)]
        # df_easy = df[(df['difficulty'] == 5) & (df['var_num'] == 100)]

        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        # 获取所有数据的 miner_num 值
        all_miner_nums = sorted(set(df_easy['miner_num'].unique()) | 
                               set(df_med['miner_num'].unique()) | 
                               set(df_hard['miner_num'].unique()))
        
        # 确定需要添加分布图的位置
        # 选择几个代表性的矿工数量点，避免图形过于拥挤
        selected_miners = [1, 3, 5, 10, 15]  # 可根据实际数据调整
        selected_miners = [m for m in selected_miners if m in all_miner_nums]
        
        # 存储各个变量的颜色和标签信息
        df_list = [df_easy, df_med, df_hard]
        label_list = ['50 variables', '100 variables', '150 variables']
        marker_list = ['x', 'o', 's']
        color_list = [colors[2], colors[1], colors[3]]
        
        # 主图中绘制统计数据
        for df, label, marker, base_color in zip(df_list, label_list, marker_list, color_list):
            # 先展开solve_rounds列表
            df_expanded = df.explode('solve_rounds')
            df_expanded['solve_rounds'] = df_expanded['solve_rounds'].astype(float)
            
            # 然后计算统计值
            stats = df_expanded.groupby('miner_num')['solve_rounds'].agg(
                ['median', lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]).reset_index()
            stats.columns = ['miner_num', 'median', 'q1', 'q3']
            n_points = len(stats)
            color_shades = [mcolors.to_rgba(base_color, alpha) 
                          for alpha in np.linspace(1, 0.3, n_points)]
            ax.plot([], [], marker=marker, color=base_color, label=label) # 图例
            
            # 保留原有的线条和点的绘制
            for i in range(len(stats)-1):
                x_fill = [stats['miner_num'][i], stats['miner_num'][i+1]]
                y1_fill = [stats['q1'][i], stats['q1'][i+1]]  # 下四分位数
                y2_fill = [stats['q3'][i], stats['q3'][i+1]]  # 上四分位数
                # ax.fill_between(x_fill, y1_fill, y2_fill, 
                #               color=color_list[i], alpha=0.2)
                ax.plot(stats['miner_num'][i:i+2], stats['median'][i:i+2], 
                       color=color_shades[i], linewidth=1.5)
                ax.plot(stats['miner_num'][i], stats['median'][i], 
                       marker=marker, color=color_shades[0])
            ax.plot(stats['miner_num'].iloc[-1], stats['median'].iloc[-1], 
                   marker=marker, color=color_shades[0])
            
            for i, (_, row) in enumerate(stats.iterrows()): # 四分位数的短横线
                ax.vlines(row['miner_num'], row['q1'], row['q3'], 
                         color=color_shades[i], alpha=0.7)
                ax.hlines(row['q1'], row['miner_num']-0.1, row['miner_num']+0.1, 
                         color=color_shades[i], alpha=0.7)
                ax.hlines(row['q3'], row['miner_num']-0.1, row['miner_num']+0.1, 
                         color=color_shades[i], alpha=0.7)
                
            # if label == "50 variables":
            #     continue
            # # 在每个数据点旁边绘制频率直方图（归一化处理）
            # for i, (_, row) in enumerate(stats.iterrows()):
            #     miner_num = row['miner_num']
                
            #     # 筛选该矿工数量下的数据
            #     data = df_expanded[df_expanded['miner_num'] == miner_num]['solve_rounds'].values
                
            #     if len(data) < 3:
            #         continue  # 数据太少则跳过
                
            #     # 获取四分位数
            #     q1 = row['q1']
            #     q3 = row['q3']
                
            #     # 只保留q1到q3之间的数据
            #     trimmed_data = data[(data >= q1) & (data <= q3)]
                
            #     if len(trimmed_data) < 3:
            #         continue  # 截断后数据太少则跳过
                
            #     # 计算直方图数据
            #     n_bins = 5 if label == "150 variables" else 5 # 直方图的箱数
            #     bin_range = (q1, q3)
            #     hist_heights, bin_edges = np.histogram(trimmed_data, bins=n_bins, range=bin_range, density=True)
                
            #     # 固定宽度，用于所有直方图
            #     fixed_width = 2.0  # 可以根据图形大小调整
                
            #     # 真正的归一化处理 - 确保每个直方图的总高度相等
            #     # 计算总面积
            #     bin_width = (bin_edges[1] - bin_edges[0])
            #     total_area = np.sum(hist_heights * bin_width)
                
            #     # 归一化到固定总高度
            #     norm_factor = 1.0 / total_area if total_area > 0 else 0
            #     norm_heights = hist_heights * norm_factor
                
            #     # 再次缩放，使得视觉效果合适
            #     scale_factor = (q3 - q1) * 0.5  # 最大高度比例，可调整
            #     scaled_heights = norm_heights * scale_factor
                
            #     # 绘制频率柱状图（只在右侧）
            #     for j in range(len(scaled_heights)):
            #         # 计算柱子高度
            #         bar_height = fixed_width  # 使用固定宽度
            #         # 计算柱子位置
            #         bar_left = miner_num 
            #         bar_bottom = bin_edges[j]  # 柱子的底部位置
                    
            #         # 创建长方形，宽度随频率变化
            #         rect = plt.Rectangle(
            #             (bar_left, bar_bottom),  # 左下角坐标
            #             bar_height * scaled_heights[j],  # 宽度根据频率变化
            #             bin_width,   # 高度固定为bin宽度
            #             facecolor=color_shades[i],
            #             alpha=0.3,
            #             edgecolor='#333333',  # 使用浅灰色边缘线
            #             linewidth=0.2  # 细线宽度
            #         )
                    
            #         # 添加到图中
            #         ax.add_patch(rect)
        
        ax.set_xlabel('Number of solvers')
        ax.set_ylabel('Solving rounds')
        ax.legend()
    
    def plot_speedup(ax:plt.Axes, df_med:pd.DataFrame, df_easy:pd.DataFrame, df_hard:pd.DataFrame, 
                     m1_med:pd.DataFrame, m1_easy:pd.DataFrame, m1_hard:pd.DataFrame):
        speedup_100 = m1_med / df_med['ave_solve_round']
        speedup_50 = m1_easy / df_easy['ave_solve_round']
        speedup_120 = m1_hard / df_hard['ave_solve_round']
        ax.plot(df_med['miner_num'], speedup_100, marker='o',color =colors[1])
        ax.plot(df_easy['miner_num'], speedup_50, marker='x',color =colors[2])
        ax.plot(df_hard['miner_num'], speedup_120, marker='s',color =colors[3])
        ax.set_xticks(range(1, 16))
        ax.set_xlabel('Number of solvers')
        ax.set_ylabel('Speed up',labelpad = 12)
        # ax.legend()
    
    def plot_efficiency(ax:plt.Axes, df_med:pd.DataFrame, df_easy:pd.DataFrame, df_hard:pd.DataFrame, 
                       m1_med:pd.DataFrame, m1_easy:pd.DataFrame, m1_hard:pd.DataFrame  ):
        speedup_100 = m1_med / df_med['ave_solve_round'] / df_med['miner_num']
        speedup_50 = m1_easy / df_easy['ave_solve_round']/ df_med['miner_num']
        speedup_120 = m1_hard / df_hard['ave_solve_round']/ df_hard['miner_num']
        ax.plot(df_med['miner_num'], speedup_100, marker='o',color =colors[1])
        ax.plot(df_easy['miner_num'], speedup_50, marker='x',color =colors[2])
        ax.plot(df_hard['miner_num'], speedup_120, marker='s',color =colors[3])
        ax.set_xlabel('Number of solvers')
        ax.set_ylabel('Efficiency')
    
    def plot_workload(ax:plt.Axes, df_med:pd.DataFrame):
        metrics = df_med[['miner_num', 'main', 'fork', 'unpub']]
        metrics = metrics.set_index('miner_num')
        colors=["#1b4332","#40916c","#74c69d","#95d5b2"]
        colors = ["#5494CE","#dc2626","#f9cc52", ] #"#0D898A"
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

    def plot_workload_by_varnum(ax2:plt.Axes):
        """绘制不同变量数下的工作量堆叠柱状图和比例堆叠柱状图"""
        # 读取数据
        file_path = pathlib.Path.cwd() / "Results\\20250211\workload_var_num.json"
        data_list = []
        with open(file_path, 'r') as f:
            jsondata_list = f.read().split('\n')[:-1]
            for jsondata in jsondata_list:
                data_list.append(json.loads(jsondata))
        df = pd.DataFrame(data_list)
        df = df.sort_values(by=["var_num", "difficulty", "miner_num"])
        
        # 数据预处理
        df['main'] = df['ave_acp_subpair_num']
        df['fork'] = df['ave_subpair_num'] - df['ave_acp_subpair_num']
        df['unpub'] = df['ave_subpair_unpubs']
        
        # 计算总工作量和比例
        df['total'] = df['main'] + df['fork'] + df['unpub']
        df['main_ratio'] = df['main'] / df['total']
        df['fork_ratio'] = df['fork'] / df['total']
        df['unpub_ratio'] = df['unpub'] / df['total']
        
        # 筛选和聚合数据
        df_filtered = df[df['difficulty'] == 5].groupby('var_num').agg({
            'main': 'mean',
            'fork': 'mean',
            'unpub': 'mean',
            'main_ratio': 'mean',
            'fork_ratio': 'mean',
            'unpub_ratio': 'mean'
        }).reset_index()
        
        # 绘图设置
        colors = ["#2b6cb0", "#dc2626", "#CCCACAFF"]  # 深蓝色、深红色、深灰色
        # colors = ["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"]
        # colors = ["#0D898A","#dc2626","#F5DEA1FF", ] #"#5494CE"
        width = 12
        
        # # 绘制工作量堆叠柱状图
        # ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)  # 先画网格，zorder设为0
        
        # ax.bar(df_filtered['var_num'], df_filtered['main'], 
        #     label='main', color=colors[0], alpha=0.8, width=width, zorder=2)  # 柱状图zorder设为2
        # ax.bar(df_filtered['var_num'], df_filtered['fork'], bottom=df_filtered['main'], 
        #     label='fork', color=colors[1], alpha=0.8, width=width, zorder=2)
        # ax.bar(df_filtered['var_num'], df_filtered['unpub'], 
        #     bottom=df_filtered['main'] + df_filtered['fork'], 
        #     label='unpublished', color=colors[2], alpha=0.8, width=width, zorder=2)
        
        # ax.set_xlabel('Number of variables')
        # ax.set_ylabel('Workload')
        # ax.legend()
        ax2.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
        # 绘制比例堆叠柱状图并添加百分比标注
        for i, row in df_filtered.iterrows():
            # 计算每个部分的中心位置
            main_center = row['main_ratio'] / 2
            fork_center = row['main_ratio'] + row['fork_ratio'] / 2
            unpub_center = row['main_ratio'] + row['fork_ratio'] + row['unpub_ratio'] / 2
            
            # 添加百分比标注，根据背景深浅选择文字颜色
            ax2.text(row['var_num'], main_center, 
                    f"{row['main_ratio']*100:.1f}%", 
                    ha='center', va='center', color='white', fontsize=11, fontweight='bold')
            ax2.text(row['var_num'], fork_center, 
                    f"{row['fork_ratio']*100:.1f}%", 
                    ha='center', va='center', color='black', fontsize=11, fontweight='bold')
            ax2.text(row['var_num'], unpub_center, 
                    f"{row['unpub_ratio']*100:.1f}%", 
                    ha='center', va='center', color='black', fontsize=11, fontweight='bold')
        
        ax2.bar(df_filtered['var_num'], df_filtered['main_ratio'], 
            label='main', color=colors[0], alpha=0.8, width=width, zorder=2)
        ax2.bar(df_filtered['var_num'], df_filtered['fork_ratio'], 
            bottom=df_filtered['main_ratio'], 
            label='fork', color=colors[1], alpha=0.8, width=width, zorder=2)
        ax2.bar(df_filtered['var_num'], df_filtered['unpub_ratio'], 
            bottom=df_filtered['main_ratio'] + df_filtered['fork_ratio'], 
            label='unpublished', color=colors[2], alpha=0.8, width=width, zorder=2)
                
        # 设置y轴为百分比格式
        # ax2.yaxis.set_major_formatter(PercentFormatter(1.0))  # 1.0表示原数据是小数形式
        ax2.set_xlabel('Number of variables')
        ax2.set_ylabel('Proportion of\nworkload')
        # ax2.legend()

    def plot_workload_per_miner(ax:plt.Axes, data:pd.DataFrame, showLegend:bool=True):
        # Extracting the adjusted metrics
        metrics_per = data[['miner_num', 'main_per', 'fork_per', 'unpub_per']]
        metrics_per = metrics_per.set_index('miner_num')
        colors=["#1b4332","#40916c","#74c69d","#95d5b2"]
        colors = ["#B2D9FF","#dc2626","#FFE4B2", ] #"#0D898A"
        colors = ["#2b6cb0", "#dc2626", "#CCCACAFF"]  # 深蓝色、深红色、深灰色
        # colors = ["#299799","#FF8784","#FFDA80", ] #"#0D898A"
        # colors=["#31572c", "#4f772d", "#90a955","#ecf39e"] "#2d6a4f","#52b788"
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)  # 先画网格，zorder设为0
        width = 1.5
        ax.bar(data['miner_num'], data['main_per'], 
               label='main', color=colors[0], alpha=0.8,width=width,zorder=2)
        ax.bar(data['miner_num'], data['fork_per'], bottom=data['main_per'], 
               label='fork', color=colors[1], alpha=0.8,width=width,zorder=2)
        ax.bar(data['miner_num'], data['unpub_per'], bottom=data['main_per'] + data['fork_per'], 
               label='unpublished', color=colors[2], alpha=0.8,width=width,zorder=2)
        # Plotting
        # metrics_per.plot(kind='bar', ax=ax, stacked=True)
        ax.set_xlabel('Number of solvers')
        ax.set_ylabel('Workload per miner',labelpad = 12)
        if showLegend:
            ax.legend()

    def plot_workload_comparison(ax:plt.Axes, df_easy:pd.DataFrame, df_med:pd.DataFrame):
        # 颜色设置
        colors = ["#2b6cb0", "#dc2626", "#CCCACAFF"]   # 深蓝色、深红色、深灰色
        
        width = 0.35  # 单个柱子的宽度
        x = np.arange(len(df_easy['miner_num']))  # 横坐标位置
        
        # 设置填充线的样式
        hatch_color = '#1B1B1BFF'  # 使用深一点的灰色
        plt.rcParams['hatch.linewidth'] = 1  # 减小填充线的粗细
        plt.rcParams['hatch.color'] = hatch_color  # 设置填充线颜色
        
        # 绘制df_med的数据（使用稀疏点状填充）
        ax.bar(x - width/2, df_med['main_per'], width,
            color=colors[0], alpha=0.9, zorder=2, hatch='///')
        ax.bar(x - width/2, df_med['fork_per'], width,
            bottom=df_med['main_per'], color=colors[1], alpha=0.9, zorder=2, hatch='///')
        ax.bar(x - width/2, df_med['unpub_per'], width, 
            bottom=df_med['main_per'] + df_med['fork_per'],
            color=colors[2], alpha=0.9, zorder=2, hatch='///')

        # 绘制df_easy的数据（使用稀疏斜线填充）
        ax.bar(x + width/2, df_easy['main_per'], width, 
            color=colors[0], alpha=0.4, zorder=2)
        ax.bar(x + width/2, df_easy['fork_per'], width, 
            bottom=df_easy['main_per'], color=colors[1], alpha=0.4, zorder=2)
        ax.bar(x + width/2, df_easy['unpub_per'], width, 
            bottom=df_easy['main_per'] + df_easy['fork_per'], 
            color=colors[2], alpha=0.4, zorder=2)
        
        # 创建自定义图例
        from matplotlib.patches import Patch
        # 工作量类型的图例（纯色）
        legend_elements1 = [
            Patch(facecolor=colors[0], label='Main', alpha=0.9),
            Patch(facecolor=colors[1], label='Fork', alpha=0.9),
            Patch(facecolor=colors[2], label='Unpublished', alpha=0.9)
        ]
        # 变量数量的图例（空心）
        legend_elements2 = [
            Patch(facecolor='white', edgecolor=hatch_color,  label='50 variables'),
            Patch(facecolor='white', edgecolor=hatch_color, hatch='///', label='100 variables'),
        ]
        
        # 添加两组图例
        ax.legend(handles=legend_elements1+legend_elements2, loc='upper right')
        
        # 设置坐标轴
        ax.set_xlabel('Number of solvers')
        ax.set_ylabel('Workload per miner')
        ax.set_xticks(x)
        ax.set_xticklabels(df_easy['miner_num'])

        
        # 添加网格
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
        # ax.legend()
        
        # 在第二和第三组柱子顶部添加标注
        # 100变量的标注
        total_height_med = df_med['main_per'].iloc[1] + df_med['fork_per'].iloc[1] + df_med['unpub_per'].iloc[1]
        ax.text(1 - width/2, total_height_med, '100 vars', 
                ha='center', va='bottom')
        
        # 50变量的标注
        total_height_easy = df_easy['main_per'].iloc[2] + df_easy['fork_per'].iloc[2] + df_easy['unpub_per'].iloc[2]
        ax.text(2, total_height_easy, '50\nvars', 
                ha='left', va='bottom')

    def plot_workload_balance(ax:plt.Axes, miner_num:int=5, var_num:int=100):
        # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250305\\214034\workload_balance.json"
        json_path = "E:\Files\gitspace\\bbb-github\Results\\20250306\\203200\\final_results.json"
        data_list = []
        with open(json_path, 'r') as f:
            for line in f:
                data_list.append(json.loads(line))
        target_data = None
        for data in data_list:
            if data['miner_num'] == miner_num and data['var_num'] == var_num:
                target_data = data
                break
        if target_data is None:
            raise ValueError(f"No data found for miner_num={miner_num}, var_num={var_num}")
        
        miner_chains = target_data['miner_chains']
        miner_mains = target_data['miner_mains']
        miner_unpubs = target_data['miner_unpubs']
        miners = sorted(miner_chains.keys(), key=int)  # 矿工ID列表按数值大小排序
        main_loads = [miner_mains[m] for m in miners]  # 主链工作量
        fork_loads = [miner_chains[m] - miner_mains[m] for m in miners]  # 分叉工作量
        unpub_loads = [miner_unpubs[m] for m in miners]  # 未发布工作量
        
        width = 0.8
        colors = ["#2b6cb0", "#dc2626", "#9ca3af"] 
        ax.bar(miners, main_loads, width, 
               label='Main chain', color=colors[0], alpha=0.8, zorder=2)
        ax.bar(miners, fork_loads, width, bottom=main_loads,
               label='Fork', color=colors[1], alpha=0.8, zorder=2)
        ax.bar(miners, unpub_loads, width, bottom=[m+f for m,f in zip(main_loads, fork_loads)],
               label='Unpublished', color=colors[2], alpha=0.8, zorder=2)
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
        ax.set_xlabel('Miner ID')
        ax.set_ylabel('Workload')
        # ax.legend()       
        ax.set_xticks(miners)

    def plot_workload_decrease(ax:plt.Axes):
        """绘制不同变量数量下的工作量减少比例"""
        # json_path = "E:\Files\gitspace\\bbb-github\Results\\20250305\\214034\workload_balance.json"
        json_path = "E:\Files\gitspace\\bbb-github\Results\\20250306\\203200\\final_results.json"
        data_list = []
        with open(json_path, 'r') as f:
            for line in f:
                data_list.append(json.loads(line))
        var_nums = sorted(set(d['var_num'] for d in data_list))
        colors = ["#2b6cb0", "#dc2626", "#40916c", "#9ca3af"]
        
        for i, var_num in enumerate(var_nums):
            var_data = [d for d in data_list if d['var_num'] == var_num]
            var_data.sort(key=lambda x: x['miner_num'])
            base_data = next(d for d in var_data if d['miner_num'] == 1)
            base_workload = sum(base_data['miner_chains'].values())
            miner_nums = []
            decrease_ratios = []
            for data in var_data:
                miner_num = data['miner_num']
                total_workload = sum(data['miner_chains'].values())
                avg_workload = total_workload / miner_num
                ratio = avg_workload / base_workload
                miner_nums.append(miner_num)
                decrease_ratios.append(ratio)
            ax.plot(miner_nums, decrease_ratios, 
                   label=f'var_num={var_num}', 
                   color=colors[i], marker='o')
        
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
        ax.set_xlabel('Number of solvers')
        ax.set_ylabel('Workload decrease ratio')
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
    
    # 创建子图并添加标签
    fig = plt.figure(figsize=(10, 8.5))
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1])
    
    # 定义标签位置
    label_x = -0.12  # 标签的x位置
    label_y = 1.01   # 标签的y位置
    
    # 创建所有子图
    axSolveRounds = fig.add_subplot(grid[0:2, 0]) 
    axSpeed1 = fig.add_subplot(grid[0:2, 1])         
    # axSpeed2 = fig.add_subplot(grid[1, 1])        
    axWorkPer = fig.add_subplot(grid[2:4, 0])        
    axWorkVarRatio = fig.add_subplot(grid[2, 1])   
    # axWorkBalance = fig.add_subplot(grid[3, 0])   
    axWorkBalance2 = fig.add_subplot(grid[3, 1])  

    # 添加标签
    axSolveRounds.text(label_x, label_y, 'a', transform=axSolveRounds.transAxes, 
                       fontsize=14, fontweight='bold')
    axSpeed1.text(label_x, label_y, 'b', transform=axSpeed1.transAxes, 
                  fontsize=14, fontweight='bold')
    # axSpeed2.text(label_x, label_y, 'c', transform=axSpeed2.transAxes, 
    #               fontsize=14, fontweight='bold')
    axWorkPer.text(label_x, label_y, 'c', transform=axWorkPer.transAxes, 
                   fontsize=14, fontweight='bold')
    axWorkVarRatio.text(label_x, label_y, 'd', transform=axWorkVarRatio.transAxes, 
                        fontsize=14, fontweight='bold')
    axWorkBalance2.text(label_x, label_y, 'e', transform=axWorkBalance2.transAxes, 
                        fontsize=14, fontweight='bold')
    
    # 设置边框颜色等
    ax_list = [axSolveRounds, axSpeed1, axWorkPer, axWorkVarRatio]  # 更新列表
    for ax in ax_list:
        for spine in ax.spines.values():
            spine.set_edgecolor('grey')
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5, zorder=0)
    
    # 读取数据
    # file_path = pathlib.Path.cwd() / "Result_Data/1226v100_50m1_20.json"
    # file_path = pathlib.Path.cwd() / "E:\Files\gitspace\\bbb-github\Results\\20250305\\214034\workload_balance_med.json"
    file_path = "E:\Files\gitspace\\bbb-github\Results\\20250306\\203200\\med_results.json"
    data_list = []
    with open(file_path, 'r') as f:
        jsondata_list = f.read().split('\n')[:-1]
        for jsondata in jsondata_list:
            data_list.append(json.loads(jsondata))
        df = pd.DataFrame(data_list)
    df = df.sort_values(by=["var_num", "difficulty", "miner_num"])
    df['main'] = df['ave_acp_subpair_num']
    df['fork'] = df['ave_subpair_num'] - df['ave_acp_subpair_num']
    df['unpub'] = df['ave_subpair_unpubs'] - df['ave_subpair_num']
    df['main_per'] = df['main'] / df['miner_num']
    df['fork_per'] = df['fork'] / df['miner_num']
    df['unpub_per'] = df['unpub'] / df['miner_num']
    df['chain_per'] = df['ave_subpair_num'] / df['miner_num']
    df['total_per'] = df['ave_subpair_unpubs'] / df['miner_num']
    df_hard = df[(df['difficulty'] == 5) & (df['var_num'] == 150)]
    df_med = df[(df['difficulty'] == 5) & (df['var_num'] == 100)]
    df_easy = df[(df['difficulty'] == 5) & (df['var_num'] == 50)]
    sr_med = df_med.groupby('miner_num')['ave_solve_round'].mean().reset_index()
    sr_easy = df_easy.groupby('miner_num')['ave_solve_round'].mean().reset_index()
    sr_hard = df_hard.groupby('miner_num')['ave_solve_round'].mean().reset_index()
    m1sr_med = df_med[df_med['miner_num'] == 1]['ave_solve_round'].mean()
    m1sr_easy = df_easy[df_easy['miner_num'] == 1]['ave_solve_round'].mean()
    m1sr_hard = df_hard[df_hard['miner_num'] == 1]['ave_solve_round'].mean()
    # plot_solve_rounds(axSolveRounds, sr_med, sr_easy)
    plot_solve_rounds_with_stats(axSolveRounds, df_easy, df_med, df_hard)
    plot_speedup(axSpeed1, sr_med, sr_easy, sr_hard, m1sr_med, m1sr_easy, m1sr_hard)
    # plot_efficiency(axSpeed2, sr_med, sr_easy, sr_hard, m1sr_med, m1sr_easy, m1sr_hard)
    # plot_workload_per_miner(axWorkPer, df_easy)
    plot_workload_comparison(axWorkPer, df_easy, df_med)
    plot_workload_by_varnum(axWorkVarRatio)  # 注意这里只传一个参数
    plot_workload_balance(axWorkBalance2, 15, 100)
    # plot_workload_per_miner(axWorkBalance, df_med, showLegend=False)
    # plot_workload_balance(axWorkBalance2, 15, 120)
    # plot_workload_decrease(axWorkBalance2)
    
    # 调整子图间距
    fig.subplots_adjust(left=0.079, bottom=0.067, right=0.98, top=0.98, hspace=0.57, wspace=0.25)
    plt.show()

if __name__ == "__main__":
    plot_solveround_workload_fig4()
