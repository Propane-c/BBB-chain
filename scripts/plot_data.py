import json
from collections import defaultdict
from pathlib import Path

import sys
import pandas as pd
sys.path.append("E:\Files\gitspace\\bbb-github")

import branchbound.bb_consensus as bb
import myplot
import myplot2

DATAFRAME = "dataframe"
MINER = "miner_num"
DIFF = "difficulty"
ATTACK = "attack"
LONG = "kb_forkrate"
SEARCH = "search_strategy"
SINGLE_EVA = "single evaluation"
LBUB = "lower upper bounds"

MAXSAT='maxsat'
TSP='tsp'
MIPLTP='miplib'

def load_json_file(filename, format):
    """根据指定的格式加载json文件为字典"""
    data_list = []
    with open(filename, 'r') as f:
        if format == SINGLE_EVA:
            data_dict = dict(json.load(f))
            return data_dict
        if format != LBUB:
            jsondata_list = f.read().split('\n')[:-1]
        else:
            jsondata_list = f.read().split('\n')
        for jsondata in jsondata_list:
            data_list.append(json.loads(jsondata))
        if format == DATAFRAME:
            df = pd.DataFrame(data_list)
            df = df.sort_values(by=["var_num", "difficulty", "miner_num"])
            return df
        if format == MINER:
            data_dict = create_data_dict_miner(data_list)
        elif format == DIFF:
            data_dict = create_data_dict_difficulty(data_list)
        elif format == LONG:
            data_dict = create_data_dict_keyforkrate(data_list)
        elif format == ATTACK:
            data_dict = create_data_dict_difficulty_attacker(data_list)
        elif format == SEARCH:
            data_dict = create_data_list_searchst(data_list)
        elif format == LBUB:
            data_dict = create_data_list_lbub(data_list)
        else: 
            raise ValueError("format error!")
        return data_dict

def create_data_list_lbub(json_data_list):
    data_dicts = [dict(js) for js in json_data_list]
    return data_dicts

def create_data_dict_keyforkrate(json_data_list):
    """对应长链仿真，keyblock的分叉率"""
    data_dict = defaultdict()
    for entry in json_data_list:
        if entry['kb_strategy'] == 'pow':
            kb_strategy = 'hashcash'
        if entry['kb_strategy'] == 'withmini':
            kb_strategy = 'w/ mini-block'
        if entry['kb_strategy'] == 'pow+withmini':
            kb_strategy = 'hashcash + mini-block'
        var_num = entry['var_num']
        difficulty = entry['difficulty']
        miner_num = entry['miner_num']
        # kb_strategy = entry['kb_strategy']
        kb_forkrate = entry['ave_kb_forkrate']
        total_kb_forkrate = entry['total_kb_forkrate']
        # mb_times = entry['mb_times']
        data_dict[(kb_strategy,difficulty)] = kb_forkrate
    return data_dict

def create_data_list_searchst(json_data_list):
    """对应长链仿真，keyblock的分叉率"""
    data_dict = defaultdict()
    data_list = []
    for entry in json_data_list:
        if entry['openblk_st'] == bb.OB_RAND and entry['openprblm_st'] == bb.OP_BEST:
            strategy = 'BFS'
        if entry['openblk_st'] == bb.OB_DEEP and entry['openprblm_st'] == bb.OP_RAND:
            strategy = 'DFS'
        if entry['openblk_st'] == bb.OB_BREATH and entry['openprblm_st'] == bb.OP_RAND:
            strategy = 'BrFS'
        if entry['openblk_st'] == bb.OB_RAND and entry['openprblm_st'] == bb.OP_RAND:
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

def create_data_dict_difficulty_attacker(json_data_list):
    """对应attack仿真"""
    data_dict = defaultdict(dict)
    for entry in json_data_list:
        # if entry['safe_thre'] in [0.003, 0.008, 0.0008, 0.0003]:
        #     continue
        if entry['difficulty'] not in [3, 5, 8, 10]:
            continue
        safe_thre = entry['safe_thre']
        difficulty = entry['difficulty']
        adversary_num = entry['adversary_num']
        ave_success_rate = entry['ave_success_rate']
        total_success_rate = entry['total_success_rate']
        ave_advrate = entry['ave_advrate']
        ave_accept_advrate = entry['ave_accept_advrate']
        mb_times = entry['mb_times']
        if 'atklog_mb' in entry:
            atklog_mb = entry['atklog_mb']
            atklog_mb = [atk for atk in atklog_mb if atk["success_num"] != 0 and atk["attack_num"] != 0]
            if len(atklog_mb) > 0:
                for atk in atklog_mb:
                    success_rate = atk["success_num"] / atk["attack_num"]
                    atk["success_rate"] = success_rate
                atklog_mb = sorted(atklog_mb, key=lambda x: x["theory"],reverse=True)
        else:
            atklog_mb = []

        data_dict[(safe_thre, adversary_num)][difficulty] = {
            'ave_success_rate': ave_success_rate,
            'total_success_rate': total_success_rate,
            'ave_advrate': ave_advrate,
            'ave_accept_advrate':ave_accept_advrate,
            'mb_times':mb_times,
            'atklog_mb':atklog_mb
        }    
    sorted_data = {}
    # 对内层字典按 difficulty 排序
    for outer_key, inner_dict in sorted(data_dict.items(), key=lambda x: x[0][1]):
        # 对内层字典按 adversary_num 排序
        sorted_inner_dict = dict(sorted(inner_dict.items(), key=lambda x: x[0]))
        sorted_data[outer_key] = sorted_inner_dict

    return sorted_data

def create_data_dict_difficulty(json_data_list):
    """对应短链仿真，difficulty作为x轴"""
    data_dict = defaultdict(dict)
    for entry in json_data_list:
        
        var_num = entry['var_num']
        difficulty = entry['difficulty']
        
        miner_num = entry['miner_num']
        ave_solve_round = entry['ave_solve_round']
        ave_subpair_num = entry['ave_subpair_num']
        ave_subpair_unpubs = entry['ave_subpair_unpubs']
        ave_acp_subpair_num = entry['ave_acp_subpair_num']
        ave_mb_forkrate = entry['ave_mb_forkrate']
        total_mb_forkrate = entry['total_mb_forkrate']
        ave_mb_growth = 1/entry['ave_mb_growth']
        # mb_times = entry['mb_times']

        data_dict[(var_num, miner_num)][difficulty] = {
            'ave_solve_round': ave_solve_round,
            'ave_subpair_num': ave_subpair_num,
            # 'ave_subpair_num': ave_subpair_unpubs,
            'ave_mb_forkrate': ave_mb_forkrate,
            'ave_subpair_unpubs': ave_subpair_unpubs,
            'ave_acp_subpair_num': ave_acp_subpair_num,
            # 'ave_subpair_unpubs': ave_subpair_num,
            'total_mb_forkrate': total_mb_forkrate,
            'ave_mb_growth':ave_mb_growth,
            # 'mb_times':mb_times
        }    
    sorted_data = {}
    for outer_key, inner_dict in sorted(data_dict.items(), key=lambda x: x[0][1]):
        # 对内层字典按 difficulty 排序
        sorted_inner_dict = dict(sorted(inner_dict.items(), key=lambda x: x[0]))
        sorted_data[outer_key] = sorted_inner_dict
    m1_subpair = {}
    m1_subunpub = {}
    m1_subacp = {}
    # 先提取出单个矿工的数据
    for (var_num, miner_num), data_single_m in sorted_data.items():
        if miner_num != 1:
            continue
        for difficulty, data in data_single_m.items():
            m1_subpair.update({difficulty: data['ave_subpair_num']})
            m1_subunpub.update({difficulty: data['ave_subpair_unpubs']})
            m1_subacp.update({difficulty: data['ave_acp_subpair_num']})
            data.update({'subpair_rate':1})
            data.update({'subunpub_rate':1})
            data.update({'subacp_rate':1})
            data_single_m.update({difficulty:data})
        sorted_data.update({(var_num, miner_num):data_single_m})
    
    # 将其他矿工的subunpub数据与单个矿工相除
    for (var_num, miner_num), data_single_m in sorted_data.items():
        if miner_num == 1:
            continue
        for difficulty, data in data_single_m.items():
            subpair_rate = data['ave_subpair_num'] / m1_subpair.get(difficulty)
            subunpub_rate = (data['ave_subpair_unpubs'] / m1_subunpub.get(difficulty))/miner_num
            subacp_rate = (data['ave_acp_subpair_num'] / m1_subacp.get(difficulty))
            data.update({'subpair_rate':subpair_rate})
            data.update({'subunpub_rate':subunpub_rate})
            data.update({'subacp_rate':subacp_rate})
            data_single_m.update({difficulty:data})
        sorted_data.update({(var_num, miner_num):data_single_m})


    return sorted_data

def create_data_dict_miner(json_data_list):
    """对应短链仿真，miner_num作为x轴"""
    df = pd.DataFrame(json_data_list)
    df = df.sort_values(by=["var_num", "difficulty", "miner_num"])
    print(df)
    return df
    # data_dict = defaultdict(dict)
    # for entry in json_data_list:
    #     if entry['openblk_st'] == bb.OB_RAND and entry['openprblm_st'] == bb.OP_BEST:
    #         strategy = 'BFS'
    #     if entry['openblk_st'] == bb.OB_DEEP and entry['openprblm_st'] == bb.OP_RAND:
    #         strategy = 'DFS'
    #     if entry['openblk_st'] == bb.OB_BREATH and entry['openprblm_st'] == bb.OP_RAND:
    #         strategy = 'BrFS'
    #     if entry['openblk_st'] == bb.OB_RAND and entry['openprblm_st'] == bb.OP_RAND:
    #         strategy = 'Rand'
    #     # difficulty = strategy
    #     var_num = entry['var_num']
    #     difficulty = entry['difficulty']
    #     miner_num = entry['miner_num']
    #     ave_solve_round = entry['ave_solve_round']
    #     ave_subpair_num = entry['ave_subpair_num']
    #     ave_subpair_unpubs = entry['ave_subpair_unpubs']
    #     ave_acp_subpair_num = entry['ave_acp_subpair_num']
    #     ave_mb_forkrate = entry['ave_mb_forkrate']
    #     total_mb_forkrate = entry['total_mb_forkrate']
    #     mb_times = entry['mb_times']
    #     kb_times = entry['kb_times']
    #     unpub_times = entry['unpub_times']
    #     fork_times = entry['fork_times']
    #     # solve_rounds = entry['solve_rounds']
    #     data_dict[(var_num, difficulty)][miner_num] = {
    #         'ave_solve_round': ave_solve_round,
    #         'ave_subpair_num': ave_subpair_num,
    #         'ave_subpair_unpubs': ave_subpair_unpubs,
    #         'ave_acp_subpair_num': ave_acp_subpair_num,
    #         'ave_mb_forkrate': ave_mb_forkrate,
    #         'total_mb_forkrate': total_mb_forkrate,
    #         'mb_times':mb_times,
    #         'kb_times':kb_times,
    #         "unpub_times":unpub_times,
    #         "fork_times":fork_times,
    #         # 'solve_rounds':solve_rounds
    #     }
    #     # print(mb_times)
    # sorted_data = {}
    # # 对外层字典按 difficulty 排序
    # for outer_key, inner_dict in sorted(data_dict.items(), key=lambda x: x[0][1]):
    #     # 对内层字典按 miner_num 排序
    #     sorted_inner_dict = dict(sorted(inner_dict.items(), key=lambda x: x[0]))
    #     sorted_data[outer_key] = sorted_inner_dict
    # m1_subpair = {}
    # m1_subunpub = {}
    # m1_subacp = {}
    # for (var, d), miner_data in data_dict.items():
    #     for miner, data in miner_data.items():
    #         if miner != 1:
    #             continue
    #         m1_subpair.update({d: data['ave_subpair_num']})
    #         m1_subunpub.update({d: data['ave_subpair_unpubs']})
    #         m1_subacp.update({d: data['ave_acp_subpair_num']})
    #         data.update({'subpair_rate':1})
    #         data.update({'subunpub_rate':1})
    #         data.update({'subacp_rate':1})
    # for (var, d), miner_data in sorted_data.items():
    #     for miner, data in miner_data.items():
    #         if miner == 1:
    #             continue
    #         subpair_rate = data['ave_subpair_num'] / m1_subpair[d]
    #         subunpub_rate = (data['ave_subpair_unpubs'] / m1_subunpub[d])/miner
    #         subacp_rate = data['ave_acp_subpair_num'] / m1_subacp[d]
    #         data.update({'subpair_rate':subpair_rate})
    #         data.update({'subunpub_rate':subunpub_rate})
    #         data.update({'subacp_rate':subacp_rate})
    #         miner_data.update({miner:data})
    #     sorted_data.update({(var, d):miner_data})
    # return sorted_data


if __name__ == "__main__":
    """载入数据"""
    format_str = DATAFRAME
    # format_str = MINER
    # format_str = DIFF
    # format_str = LONG
    # format_str = ATTACK
    # format_str = SEARCH
    # format_str = SINGLE_EVA
    # format_str = LBUB
    # data = load_json_file(Path.cwd()/"Result_Data"/ "1210short_data测试不同st.json", format_str)Results\20240104\Results\
    # data = load_json_file(Path.cwd()/"Result_Data\\tsp solving process.json", format_str)
    # data = load_json_file(Path.cwd()/"Result_Data\\0131tspm135mbtimes2.json", format_str)
    # data = load_json_file(Path.cwd()/"Result_Data\m1d5vmaxsatevaluation results.json", format_str)
    data = load_json_file(Path.cwd()/"Result_Data\\1226v100_50m1_20.json", format_str)
    # data = load_json_file(Path.cwd()/"Result_Data\maxsatfig3\m1d5vmaxsatevaluation results.json", format_str)
    # print(data)Result_Data\1226v100m1_20.jsonResults\
    """画图"""
    # myplot.plot_bar_chart_mbkbforkrate_diff_strategy(data, 'kb')
    if format_str == DIFF:
        # myplot.plot_line_chart_difficulty_as_xlabel(data, "Mini-block fork rate", [-0.05,0.1], 'ave_mb_forkrate')
        # myplot.plot_line_chart_difficulty_subpair_subunpub(data, "Increased workload brought about by distribution", [0.8,1.8], False)
        # myplot.plot_line_chart_difficulty_subpair_subunpub(data, "Increased workload brought about by distribution", [0,2.3], True)
        # myplot.plot_line_chart_difficulty_as_xlabel(data, "Number of solving round for a single problem ", [50,3000], 'ave_solve_round')
        # myplot.plot_line_chart_difficulty_as_xlabel(data, "Mini-block fork rate", [-0.05, 0.3], 'total_mb_forkrate')
        myplot.plot_line_chart_difficulty_as_xlabel(data, "Mini-block growth rate", [-0.05,0.3], 'ave_mb_growth')
        ...
    elif format_str == MINER:
        # myplot.plot_line_chart_miner_workload(data, "Increased workload brought about by distribution", [0,2000], True)
        # myplot.plot_line_chart_miner_as_xlabel(data, "The stale-rate of mini-blocks", [-0.02,0.3], 'ave_mb_forkrate')
        # myplot.plot_line_chart_miner_as_xlabel2(data, "Increased workload on chain", [700, 5000], 'ave_subpair_num')
        # myplot.plot_line_chart_miner_as_xlabel2(data, "Increased workload unpublished", [700, 7000], 'ave_subpair_unpubs')
        # myplot.plot_line_chart_miner_as_xlabel2(data, "Increased workload on main chain", [0, 3000], 'ave_acp_subpair_num')
        # myplot.plot_line_chart_miner_as_xlabel(data, "Number of solving round for a single problem ", [50,2000], 'ave_solve_round')
        # myplot.plot_line_chart_miner_as_xlabel2(data, "Mini-block fork rate", [-0.2,1], 'total_mb_forkrate')
        mb_blocktimes=[]
        # print(data)
        # print([k for k in data.keys()])
        d = [3,5,7,9]
        miner_num = 1
        v = ['tsp']
        # for d in difficulties:
        #     for v in var_nums:
        mb_blocktimes = data[(data['var_num'].isin(v)) & (data['difficulty'] .isin(d) ) & 
                            (data['miner_num'] == miner_num)]['mb_times'].reset_index(0,drop=True)
        for mb_blocktime in mb_blocktimes.values:
            print(len(mb_blocktime))
        print(mb_blocktimes)
                # print(len(mb_blocktime), max(mb_blocktime))
        # mb_blocktimes.append(mb_blocktime)
                # kb_blocktime = data[(v,d)][miner_num]['solve_rounds']
                # print(len(kb_blocktime), max(kb_blocktime))
                # mb_blocktimes.append(kb_blocktime)
            # mb_blocktimes.append(mb_blocktime)
        # myplot.plot_keyblock_time_pdf(mb_blocktimes, var_nums,'Inter-block time of key-block',200,0.1)
        myplot.plot_block_time_pdf(mb_blocktimes, d, 'Inter-block time of mini-blocks')
    elif format_str == ATTACK:
        # myplot.plot_bar_chart_attack(data, "Proportion of adversary blocks in all blocks", 'ave_advrate',[0,0.003])
        # myplot.plot_bar_chart_attack(data, "Proportion of adversary blocks in accepted blocks",'ave_accept_advrate', [0,0.0006])
        # myplot.plot_bar_chart_attack(data, "Average success rate of plagiarism", 'ave_success_rate',[0,0.003])
        # myplot.plot_bar_chart_attack(data, "Success rate of plagiarism", 'total_success_rate',[0,0.003])
        # myplot.plot_line_chart_attaker(data, "Proportion of adversary blocks in all blocks", [0,0.6], 'ave_advrate')
        # myplot.plot_line_chart_attaker(data, "Proportion of adversary blocks in accepted blocks", [0,0.3], 'ave_accept_advrate')
        # myplot.plot_line_chart_attaker(data, "Average success rate of plagiarism", [0,0.6], 'ave_success_rate')
        # myplot.plot_line_chart_attaker(data, "Success rate of plagiarism", [0,0.6], 'total_success_rate')
        # myplot.plot_atklog(data[(0.001, 1)][10]['atklog_mb'])
        myplot.plot_security_fig6(data[(0.001, 1)][10]['atklog_mb'])
        ...
    elif format_str == LONG:
        myplot.plot_bar_chart_kb_forkrate(data,"Key-block fork rate")
    elif format_str == SEARCH:
        # myplot.plot_bar_chart_diff_search(data, "total_mb_forkrate","Mini-block fork rate")
        # myplot.plot_bar_chart_diff_search(data, "ave_solve_round","Average number of rounds to solve a single problem")
        # myplot.plot_bar_chart_diff_search(data, "ave_subpair_num","Workload on blockchain")
        # myplot.plot_bar_chart_diff_search(data, "ave_subpair_unpubs","Total combined workload of on-chain and unpublished efforts")
        # myplot.plot_bar_chart_diff_search(data, "ave_acp_subpair_num","Workload on main chain")
        myplot.draw_radars(data)
    elif format_str ==LBUB:
        # myplot.plot_relaxed_sulotions(data)
        myplot2.plot_bounds_fig3(data, MAXSAT)

    elif format_str ==DATAFRAME:
        # myplot2.plot_mbtime_grow_fig5()
        myplot2.plot_solveround_workload_fig4()  # 使用默认路径
        # myplot2.plot_security_fig6()

    

# 两pair链式pair，m=2
# prob = 0.01
# unpub: 0.2512616261024064(1.0050464) 4.0112, 1.00865
# fk: 0.0025126162610240687 0.002792181890706023
# prob = 0.05
# pnums: 0.2564538364604931(1.0258152) 4.0523, 1.0252
# fk: 0.012822691823024675 0.012906250771166992
# prob = 0.1
# pnums: 0.2624289254993441(1.0515472) 4.103, 1.04365
# fk: 0.026288686397434015 0.025103582744333414
# prob = 0.2
# pnums: 0.27434842249657077(1.0973688) 4.232666666666667 1.060222
# fk:0.05464142661179694  0.05496928650181131
# prob = 0.3
# pnums: 0.284958273967026(1.13983308) 4.3718, 1.08125,
# fk: 0.08288310604518627 0.08504506153071961

# 两pair链式pair，m=3
# prob = 0.01
# pnums: 0.18885763437207445 4.018933333333333, 1.8008666666666666, 
# fk: 0.004464421901635422 0.00431463657484235
# prob = 0.05
# pnums: 0.1939904847341037 4.088133333333333,  1.8335333333333332
# fk: 0.022791944274061202 0.021654056874510826
# p:  0.1
# pnums: 0.2009420423416027 4.1852, 1.8964666666666667
# fk: 0.047034747378861215 0.045831875576476576,
#  p:  0.2
# pnums: 0.21559250114311854 4.404533333333333, 1.9813333333333334
# fk: 0.10040646945415685 0.09428493795851825
# p:  0.3
# pnums: 0.27627702015062083 4.6898333333333335, 2.044
# fk: 0.158018015008709 0.15611814345991562,

# 三pair, m=2
# prob:  0.01
# pnums: 0.37614385156444047(2.2568628) 2.259203681472589,
# fk: 0.0018863959394056248 0.0016971149046620744
# prob:  0.05
# pnums: 0.37956114190606427(2.277366) 2.2755
# fk: 0.009619603272953423 0.009148858869769131,
# prob:  0.1
# pnums: 0.38382956416524544(2.30297736) 2.30725,
# fk: 0.019772442015040585 0.02040816326530612,
# prob:  0.2
# pnums: 0.39120052837473973(2.3472) 2.3492
# fk: 0.04195195285271554 0.03916984274413093
# prob:  0.3
# pnums: 0.3963779451029223(2.37827) 2.3778
# fk: 0.06761455555031246 0.06150284677469812

# 三pair，m=3
# p:  0.01
# pnums: 0.18885763437207445 6.02306820454697, 4.125741716114407
# fk: 0.0032261169919394045 0.00312354622183824
# p:  0.05
# pnums: 0.1939904847341037 6.119666666666666, 4.182733333333333
# fk: 0.0164153610901637 0.016780283167278448
# p:  0.1
# fk: 0.03363416091735942 0.03297553428101731
# p:  0.2
# fk: 0.07101436784931667  0.06890130353817504
# p:  0.3
# fk: 0.11358750051166021 0.10873440285204991
