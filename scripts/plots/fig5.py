import json
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import find_peaks, peak_prominences

def plot_mbtime_grow_fig5():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(10, 8))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.5, 1.5, 2],width_ratios=[1.5, 1, 1.5,1]) # 3 rows, 2 columns
    # sns.set(style="whitegrid")
    # colors = ["#B96666","#78BCFF","#66A266","#F2A663","#BEA9E9"]
    colors = ["#FF8283", "#0D898A","#f9cc52","#5494CE", "#9467bd"]

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
        ax.legend(title="Solver num")
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
        leg = ax.legend(title="Difficulty", loc = "lower right", bbox_to_anchor=(1, 0.05))
        ax.set_ylabel('Number of solvers')
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
        ax.legend(title="Solver num")
        ax.grid(which='both', color='#dddddd', linestyle='-', linewidth=0.5)
        # sns.boxplot(x='difficulty', y='grow_proc', data=data_df, ax=ax)
    
    def plot_grow_proc(ax:plt.Axes, data_df:pd.DataFrame):
        # 确保 mb_times 是列表而不是字符串
        new_data = data_df.copy()
        
        times = new_data['mb_times'].iloc[0]  # 获取第一行的数据
        mean_time = np.mean(times)
        
        # 绘制区块时间序列
        ax.plot(range(len(times)), times, alpha=0.8, color="#f9cc52")
        ax.fill_between(range(len(times)), times, 
                label="Block time", alpha=0.8, color="#f9cc52")
        
        # 绘制平均值线
        ax.hlines(mean_time, 0, len(times),
                  linestyles='--', linewidth=2, 
                  color="orange", label="Mean")
        
        ax.legend()
        ax.set_xlabel('Blocks')
        ax.set_ylabel('Block time')
    data_list = []
    # with open(pathlib.Path("Result_Data\\0207tspm125mbtimes.json"), 'r') as f:
    with open(pathlib.Path("Result_Data\mbtimes\\0407tspm125mbtimes.json"), 'r') as f:    
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
    data_list = []
    with open(pathlib.Path("Result_Data\\mbtimes\\m5d5vburma14.json"), 'r') as f:
        jsondata_list = f.read().split('\n')
        for jsondata in jsondata_list:
            data_list.append(json.loads(jsondata))
        proc_df = pd.DataFrame(data_list)
        proc_df = proc_df.sort_values(by=["miner_num","difficulty" ])
    plot_grow_proc(axGrowProc, proc_df[(proc_df['miner_num'] == 5) & (proc_df['difficulty'].isin([5]))].copy())
    
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

    # if SAVE:
    #     plt.savefig(SAVE_PREFIX + "\\mbtimes.svg", dpi=220)
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
    ax.legend(loc='upper right', title = "Difficulty")

if __name__ == "__main__":
    plot_mbtime_grow_fig5()