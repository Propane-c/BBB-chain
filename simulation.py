import gc
import json
import logging
import multiprocessing as mp
import os
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import psutil
from pympler import asizeof

from branchbound import bb_consensus as bb
from data import lpprblm
from background import Background
from environment import Environment
from evaluation import EvaResult

# 不同矿工数量＼不同难度值下求解20个变量的问题的平均求解速度
logger = logging.getLogger(__name__)

def new_environment(background:Background, lp:lpprblm, miner_num,  
        difficulty, adversary_num, safe_thre = 1, solve_prob = 0.5,
        opblk_st:str = bb.OB_RAND, opprblm_st:str = bb.OP_RAND):
    """建立问题池中只有一个问题的环境"""
    background.set_genesis_prblm(lp)
    background.init_gases([lp])
    background.set_miner_num(miner_num)
    background.set_bb_difficulty(difficulty)
    background.reset_block_number()
    background.reset_key_prblm_number()
    background.set_safe_thre(safe_thre)
    background.set_solve_prob(solve_prob)
    background.set_openblock_strategy(opblk_st)
    background.set_openprblm_strategy(opprblm_st)
    return Environment(background, adversary_num, 50, 'equal', 'F', None, {})

def get_prblm_pool(pool_size, var_num, method = None, pool_save_path = None):
    """获取问题池, 可以指定pool_size和var_num随机产生, 也可以从"""
    m = 'rand' if method is None else method
    if m == 'load':
        # 读取问题池
        print(f"Loading problem pool--{mp.current_process().name}")
        pool_path = (Path.cwd()/"Problem Pools"/"01"/f"{var_num}vars.json")
        # pool_path = Path(".\\testTSP\problem poolburma14.json")
        # pool_path = Path.cwd() / "Problem Pools" / f"20250103\\{var_num}vars.json"
        # pool_path = Path.cwd()/ "Problem Pools" / "1109\problem pool1109_1511.json"
        # pool_path = Path.cwd()/ "Problem Pools" / "1116\problem pool1116_105207.json"
        # pool_path = Path.cwd() / "Problem Pools" / "SPOT" / f"Generated2\\{var_num}_1.json"
        prblm_pool = lpprblm.load_prblm_pool_from_json(pool_path, pool_save_path)
    elif m == 'rand':
        prblm_pool = lpprblm.prblm_pool_generator(pool_size, var_num, lpprblm.ZERO_ONE)
        lpprblm.save_prblm_pool(prblm_pool, pool_save_path, lpprblm.ZERO_ONE)
    return prblm_pool


@dataclass
class res_intermediate_full:
    """
    记录仿真过程中的仿真数据中间结果
    """
    var_num:int
    difficulty:int
    miner_num:int
    adversary_num:int
    safe_thre:float
    solve_prob:float
    openblk_st:str
    openprblm_st:str
    gas:int
    # 缓存中间结果
    solve_rounds: list = field(default_factory=list)
    subpair_nums: list = field(default_factory=list)
    acp_subpair_nums: list = field(default_factory=list)
    subpair_unpubs: list = field(default_factory=list)
    mb_forkrates: list = field(default_factory=list)
    mb_nums: list = field(default_factory=list)
    accept_mb_nums: list = field(default_factory=list)
    mb_growths: list = field(default_factory=list)
    ## 偷答案
    attack_nums: list = field(default_factory=list)
    success_nums: list = field(default_factory=list)
    success_rates: list = field(default_factory=list)
    atklog_depth: dict = field(default_factory=dict)
    atklog_mb: list = field(default_factory=list)
    ## 攻击者区块占得数量和比例
    advblock_nums: list = field(default_factory=list)
    accept_advblock_nums: list = field(default_factory=list)
    adv_rates: list = field(default_factory=list)
    accept_adv_rates: list = field(default_factory=list)
    ## 动态平均值
    dyn_avesr: list = field(default_factory=list)
    dyn_avespn: list = field(default_factory=list)
    dyn_avembfr: list = field(default_factory=list)
    # 结果
    ave_solve_round: float = 0.0
    ave_subpair_num: float = 0.0
    ave_acp_subpair_num: float = 0.0
    ave_subpair_unpubs: float = 0.0
    ave_mb_forkrate: float = 0.0
    total_mb_forkrate: float = 0.0
    ave_mb_growth: float = 0.0
    ## 偷答案成功概率
    ave_success_rate: float = 0.0
    total_success_rate: float = 0.0
    ## 攻击者区块的比例
    ave_advrate: float = 0.0
    ave_accept_advrate: float = 0.0
    total_advrate: float = 0.0
    total_accept_advrate: float = 0.0
    ## 出块时间
    mb_times: list = field(default_factory=list)
    kb_times: list = field(default_factory=list)
    unpub_times: list = field(default_factory=list)
    fork_times: list = field(default_factory=list)
    grow_proc: list = field(default_factory=list)
    ## 求解误差
    solutions_by_bbb: list = field(default_factory=list)
    solutions_by_pulp: list = field(default_factory=list)
    solution_errors: list = field(default_factory=list)
    ave_solution_error: float = 0.0
    solve_rate:float = 0.0
    not_solve_rate:float = 0.0
    # 每个矿工的工作量
    miner_chains:dict = field(default_factory=dict)
    miner_mains:dict = field(default_factory=dict)
    miner_unpubs:dict = field(default_factory=dict)

@dataclass
class res_lite:
    """轻量级仿真结果，代表图中的一个点"""
    var_num:int
    difficulty:int
    miner_num:int
    adversary_num:int
    safe_thre:float
    solve_prob:float
    openblk_st:str
    openprblm_st:str
    gas:int
    # 结果
    ave_solve_round:float
    ave_subpair_num:float
    ave_acp_subpair_num:float
    ave_subpair_unpubs:float
    ave_mb_forkrate:float
    total_mb_forkrate:float
    ave_mb_growth:float
    ## 偷答案成功概率
    ave_success_rate:float
    total_success_rate:float
    atklog_depth:dict
    atklog_mb:list
    ## 攻击者区块的比例
    ave_advrate:float
    ave_accept_advrate:float
    total_advrate:float
    total_accept_advrate:float
    ## 出块时间
    mb_times:list
    kb_times:list
    unpub_times:list
    fork_times:list
    grow_proc:list
    ## 求解误差
    ave_solution_error:float
    solve_rate:float
    not_solve_rate:float
    # 每个矿工的工作量
    miner_chains:dict
    miner_mains:dict
    miner_unpubs:dict

@dataclass
class res_final:
    """汇总所有仿真结果"""
    data_list: list[res_lite]
    
    
def med_res_to_lite_res(med_res:res_intermediate_full):
    lite_fields = {field.name for field in fields(res_lite)}
    med_res_dict = {key: value for key, value in med_res.__dict__.items() if key in lite_fields}
    return res_lite(**med_res_dict)

def simudata_collect_to_json(sd_collect:res_final, file_path, file_name=None):
    print("saving.....")
    if file_name is None:
        file_name = f'final_results_{time.strftime("%H%M%S")}.json'
    with open(file_path / file_name, "w+") as f:
        for res_lite in sd_collect.data_list:
            res_lite_dict = asdict(res_lite)
            filtered_dict = {
                key: value
                for key, value in res_lite_dict.items()
                if not (
                    (isinstance(value, (int, float)) and value == 0) or
                    (isinstance(value, (list, dict)) and not value)
                )
            }
            p_json = json.dumps(filtered_dict)
            f.write(p_json + '\n')

def short_simulation(background:Background, repeat_num:int,pool_size:int, var_num:int, 
                     difficulties:list = None, miner_nums:list = None,adversary_num = None, 
                     prblm_pool_method:str = None,recBlockTimes:bool = False, safe_thre = 1,
                     solve_prob = 0.5,opblk_st:str = bb.OB_RAND, opprblm_st:str = bb.OP_RAND, gas = 10000):
    # 结果保存路径
    ROOT_PATH = background.get_result_path()
    if adversary_num >0:
        RESULT_PATH = ROOT_PATH / f"attack_v{var_num}m{miner_nums}d{difficulties}a{adversary_num}-{mp.current_process().pid}"
    else:
        RESULT_PATH = ROOT_PATH / f"short_g{gas}v{var_num}m{miner_nums}d{difficulties}--{time.strftime('%H%M%S')}-{mp.current_process().pid}"
    RESULT_PATH.mkdir(parents=True)
    CACHE_PATH = RESULT_PATH / 'result_cache'
    CACHE_PATH.mkdir(parents=True)
    difficulties = [5, 7, 9] if difficulties is None else difficulties
    miner_nums = [1, 3, 5, 7, 10, 15, 20, 30] if miner_nums is None else miner_nums
    prblm_pool_method = 'rand' if prblm_pool_method is None else prblm_pool_method
        
    background.set_var_num(var_num)
    background.set_total_gas(gas)
    prblm_pool = get_prblm_pool(pool_size, var_num, prblm_pool_method, RESULT_PATH)
    lite_res = res_final([])

    for difficulty in difficulties:
        logger.info(f"var_num:{var_num}, difficulty: {difficulty}")
        for miner_num in miner_nums:
            # 保存错误日志
            error_dir = f"errorlogs_v{var_num}d{difficulty}m{miner_num}a{adversary_num}"
            ERROR_PATH = RESULT_PATH / error_dir
            ERROR_PATH.mkdir(parents=True)
            med_reses:list[res_intermediate_full] = []
            med_res = res_intermediate_full(
                var_num=var_num,
                difficulty=difficulty,
                miner_num=miner_num,
                adversary_num=adversary_num,
                safe_thre=safe_thre,
                solve_prob=solve_prob,
                openblk_st=opblk_st,
                openprblm_st=opprblm_st,
                gas=gas
            )
            # 进行迭代仿真
            for rep_cnt in range(repeat_num): 
                short_sim_iter(
                    rep_cnt, pool_size, miner_num, difficulty, var_num, adversary_num, med_res, background, 
                    prblm_pool, recBlockTimes, ERROR_PATH, safe_thre, solve_prob, opblk_st, opprblm_st
                )
            cal_average_save_med_res(med_res, lite_res, RESULT_PATH, CACHE_PATH)
    simudata_collect_to_json(lite_res, RESULT_PATH)
    
def merge_intermediate_res(data_list: list[res_intermediate_full]) -> res_intermediate_full:
    """
    合并多个IntermediateRes实例，将列表字段逐位求平均，生成新的IntermediateRes实例。
    """
    merged_data = {} 
    # Process all fields
    for field in fields(res_intermediate_full):
        field_name = field.name
        field_type = field.type
        
        if field_name == "solution_errors":  # Handle list fields
            merged_data[field_name] = np.mean(
                [getattr(instance, field_name) for instance in data_list], axis=0
            ).tolist()
        else:  # For non-list, non-numeric fields, take the value from the first instance
            merged_data[field_name] = getattr(data_list[0], field_name)
    
    return res_intermediate_full(**merged_data)


def evn_exec_with_error(prblm_num:int, Evn:Environment, evn_exec_done:threading.Event, 
                        continue_flag:threading.Event, ERROR_PATH):
    try:
        Evn.exec(evn_exec_done, continue_flag, ERROR_PATH)
    except Exception:
        err_path = ERROR_PATH / f"exce_error_p{prblm_num}"
        err_path.mkdir(parents=True)
        with open(err_path / "error_info.txt", "w+") as f:
            traceback.print_exc(file=f)
        print("Error encountered in evn_exec_thread. Skipping...")
        # Evn.miners[0].local_chain.ShowStructureWithGraphviz(None, None,
            # graph_path = ERROR_PATH, graph_title = f"exce_error_iter{iter_num}")
        # Evn.miners[0].local_chain.printchain2txt(err_path)
        evn_exec_done.set()

def env_evec_with_monitor(Evn:Environment, process_id, iter_num, process:psutil.Process, 
                          ERROR_PATH, continue_flag:threading.Event):
    """
    内存实时监测模块
    """
    memory_thre = 10 * 1024 * 1024 * 1024
    exec_done = threading.Event()
    # 启动环境exec线程，并设置为守护线程，确保能够第一时间结束线程
    exec_thread = threading.Thread(target=evn_exec_with_error, args=(iter_num, Evn, exec_done, continue_flag, ERROR_PATH))
    exec_thread.daemon = True
    exec_thread.start()
    while not exec_done.is_set():
        memory_usage = process.memory_info().rss
        if memory_usage < memory_thre:
            time.sleep(1)
            continue
        evn_size = asizeof.asizeof(Evn)
        record_memory_exceed_err(iter_num, Evn, evn_size, process_id, memory_usage, memory_thre, ERROR_PATH)
        continue_flag.set()  # 通知主线程继续循环
        print("Memory usage exceeded. Skipping...")
        time.sleep(2)
        return
    exec_done.clear()
    return

def record_memory_exceed_err(iteration, outer_evn:Environment, outer_evn_size, 
                             current_process_id, current_memory_usage, memory_threshold, ERROR_PATH):
    """记录超出内存的仿真参数"""
    with open(ERROR_PATH / f"memory_exceeded_iter{iteration}.txt", "a") as f:
        f.write(f"PID{current_process_id} Memory usage =-0{current_memory_usage} exceeded {memory_threshold} bytes.\n")
        f.write(f"Evrionment memory size: {outer_evn_size}\n")
        for miner in outer_evn.miners:
            f.write(f"Miner{miner.miner_id} memory size: {asizeof.asizeof(miner)}\n")
            f.write(f"    Chain{miner.miner_id} memory size: {asizeof.asizeof(miner.local_chain)}\n")
            f.write(f"    Consensus{miner.miner_id} memory size: {asizeof.asizeof(miner.consensus)}\n")
        f.write(str(outer_evn.background.get_genesis_prblm()))
    with open(ERROR_PATH / "a_hard_prblms_memory.json", "a+") as f:
        prblm = outer_evn.background.get_genesis_prblm()
        p_json = json.dumps({'c': prblm.c.tolist(),'G_ub':prblm.G_ub.tolist(),'h_ub':prblm.h_ub.tolist()})
        f.write(p_json + '\n')

def short_sim_iter(rep_cnt, pool_size, miner_num, difficulty, var_num, 
                   adversary_num, sd_spec, background, prblm_pool, 
                   recBlockTimes, ERROR_PATH, safe_thre = 1, solve_prob = 0.5,
                   opblk_st:str = bb.OB_RAND, opprblm_st:str = bb.OP_RAND):
    # 开始迭代
    current_process_id = os.getpid()
    current_process = psutil.Process(current_process_id)
    for i in range(pool_size):
        # 新建环境并运行
        eva_res = None
        try:
            print(f"prblm_count:{i}")
            # 新建环境对象
            Evn = new_environment(background, prblm_pool[i], miner_num, difficulty, adversary_num, safe_thre, 
                                solve_prob, opblk_st, opprblm_st)
            print(f"PID{mp.current_process().pid}: m{Evn.miner_num}d{difficulty}", 
                f"v{var_num}ad{adversary_num}--repeat--{rep_cnt}--prblm_count{i}", )
            continue_flag = threading.Event()
            # 启动内存监测线程
            monitor_thread = threading.Thread(target=env_evec_with_monitor,
                args=(Evn, current_process_id, i, current_process, ERROR_PATH, continue_flag))
            monitor_thread.start()
            monitor_thread.join()
            # 如果监测超出内存限制，执行continue
            if continue_flag.is_set():
                # 内存超出阈值，主线程执行continue，跳过当前迭代
                print(f"iter{i} cotinue....")
                continue_flag.clear()
                print(f"iter{i} gc collecting....")
                del Evn
                gc.collect()
                time.sleep(10)
                continue
            eva_res = Evn.view(ERROR_PATH = ERROR_PATH)

        except Exception:
            # 遇到错误，跳过当前迭代并保存错误信息
            with open(ERROR_PATH / f"error_iter{i}.txt", "w+") as f:
                traceback.print_exc(file = f)
            print("Error encountered. Skipping...")
            # sys.exit()
            continue
        # 如果有内容则更新中间结果
        if eva_res is not None and len(eva_res.mb_nums) > 0:
            record_med_res(sd_spec, eva_res, recBlockTimes)


def record_med_res(med_res:res_intermediate_full, eva_res:EvaResult,recBlockTimes:bool):
    """
    记录中间结果 
        :各个keyblock的求解时间solve_rounds
        :各个keyblock的子问题数量subpair_nums
        :各个keyblock的miniblock数量mb_nums
        :各个keyblock被接受的miniblock数量accept_mb_nums
        :各个keyblock下的miniblock分叉率mb_forkrates
    """
    # if len(eva_res.mb_times)==0:
    #     return simudata
    # 求解轮数
    for sr in eva_res.solve_rounds.values():
        med_res.solve_rounds.append(sr) # 求解轮数
    # 子问题对总数
    for spn in eva_res.subpair_nums.values():
        med_res.subpair_nums.append(spn)
    for acp_spn in eva_res.acp_subpair_nums.values():
        med_res.acp_subpair_nums.append(acp_spn)
    # 包括未发布的子问题对总数
    for spn_unpub in eva_res.subpair_unpub.values():
        med_res.subpair_unpubs.append(spn_unpub)
    # miniblock分叉率
    for mbfr in eva_res.mb_forkrates.values():
        med_res.mb_forkrates.append(mbfr)
    # miniblock数量
    for mb_num in eva_res.mb_nums.values():
        med_res.mb_nums.append(mb_num)
    # 接受链上miniblock的数量
    for acp_mb_num in eva_res.accept_mb_nums.values():
        med_res.accept_mb_nums.append(acp_mb_num)
    for mb_growth in eva_res.mb_growths.values():
        med_res.mb_growths.append(mb_growth)
    # 偷答案
    if eva_res.attack_num > 0:
        med_res.attack_nums.append(eva_res.attack_num)
        med_res.success_nums.append(eva_res.success_num)
        med_res.success_rates.append(eva_res.success_rate)
        for depth in eva_res.atklog_depth.keys():
            if depth not in med_res.atklog_depth.keys():
                med_res.atklog_depth[depth] = {"attack_num":0,"success_num":0,"success_rate":0}
            med_res.atklog_depth[depth]["attack_num"] += eva_res.atklog_depth[depth]["attack_num"]
            med_res.atklog_depth[depth]["success_num"] += eva_res.atklog_depth[depth]["success_num"]
        for bname in  eva_res.atklog_bname.keys():
            med_res.atklog_mb.append(eva_res.atklog_bname[bname])
        for advblk_num in eva_res.advblock_nums.values():
            med_res.advblock_nums.append(advblk_num)
        for acp_advblk_num in eva_res.accept_advblock_nums.values():
            med_res.accept_advblock_nums.append(acp_advblk_num)
        for advrate in eva_res.adv_rates.values():
            med_res.adv_rates.append(advrate)
        for acp_advrate in eva_res.accept_adv_rates.values():
            med_res.accept_adv_rates.append(acp_advrate)
    for se in eva_res.solution_errors:
        med_res.solution_errors.append(se)
    for sol in eva_res.solutions_by_bbb:
        med_res.solutions_by_bbb.append(sol)
    for sol in eva_res.solutions_by_pulp:
        med_res.solutions_by_pulp.append(sol)

    # 出块时间
    if recBlockTimes:
        # miniblock出块时间
        med_res.mb_times.extend(eva_res.mb_times)
        # keyblock出块时间
        med_res.kb_times.extend(eva_res.kb_times)
        med_res.unpub_times.extend(eva_res.unpub_times)
        med_res.fork_times.extend(eva_res.fork_times)
        med_res.grow_proc.extend(eva_res.mb_grow_proc)
    # 矿工工作量
    for miner in eva_res.miner_chains:
        med_res.miner_chains[miner] = med_res.miner_chains.get(miner, 0) + eva_res.miner_chains.get(miner, 0)
        med_res.miner_mains[miner] = med_res.miner_mains.get(miner, 0) + eva_res.miner_mains.get(miner, 0)
        med_res.miner_unpubs[miner] = med_res.miner_unpubs.get(miner, 0) + eva_res.miner_unpubs.get(miner, 0)
    # 动态平均值（收敛情况）
    # if len(simudata.solve_rounds) > 0:
    #     avesr = sum(simudata.solve_rounds)/len(simudata.solve_rounds)
    #     avespn = sum(simudata.subpair_nums)/len(simudata.subpair_nums)
    #     simudata.dyn_avesr.append(avesr)
    #     simudata.dyn_avespn.append(avespn)
    return med_res

def cal_average(data_list:list):
    if len(data_list) == 0:
        raise ValueError("divided by zero")
    return sum(data_list)/len(data_list)

def cal_average_save_med_res(med_res:res_intermediate_full, res_collect:res_final, result_path, cache_path):
    """
    计算并保存对特定var_num、difficulty、miner_num仿真结果
    """
    m = med_res.miner_num
    d = med_res.difficulty
    v = med_res.var_num
    # logger.info(f"miner_num: {m}, diffculty: {d}\n")
    if len(med_res.solve_rounds) > 0:
        # 平均求解总轮数
        med_res.ave_solve_round = cal_average(med_res.solve_rounds)
        # 平均子问题对数量
        med_res.ave_subpair_num = cal_average(med_res.subpair_nums)
        med_res.ave_acp_subpair_num = cal_average(med_res.acp_subpair_nums)
        # 平均子问题对数量（含未发布）
        med_res.ave_subpair_unpubs = cal_average(med_res.subpair_unpubs)
        # miniblock孤块率
        med_res.ave_mb_forkrate = cal_average(med_res.mb_forkrates)
        med_res.total_mb_forkrate = (sum(med_res.mb_nums)-sum(med_res.accept_mb_nums)) / sum(med_res.mb_nums)
        # mb growth rate
        med_res.ave_mb_growth = cal_average(med_res.mb_growths)
        if len(med_res.success_rates)>0:
            # 平均攻击成功概率
            med_res.ave_success_rate = cal_average(med_res.success_rates)
        if sum(med_res.attack_nums) > 0:
            # 攻击成功概率
            med_res.total_success_rate = sum(med_res.success_nums)/sum(med_res.attack_nums)
            for mb_depth in med_res.atklog_depth.keys():
                success_num = med_res.atklog_depth[mb_depth]["success_num"]
                attack_num = med_res.atklog_depth[mb_depth]["attack_num"]
                med_res.atklog_depth[mb_depth]["success_rate"] = success_num / attack_num
        if len(med_res.accept_adv_rates)>0:
            # 攻击者区块占的比例
            med_res.ave_advrate = cal_average(med_res.adv_rates)
            med_res.ave_accept_advrate = cal_average(med_res.accept_adv_rates)
            med_res.total_advrate = sum(med_res.advblock_nums)/sum(med_res.mb_nums)
            med_res.total_accept_advrate = sum(med_res.accept_advblock_nums)/sum(med_res.accept_mb_nums)
        # 求解误差
        if len(med_res.solution_errors)>0:
            med_res.ave_solution_error = cal_average(med_res.solution_errors)
            med_res.solve_rate = med_res.solution_errors.count(0) / len(med_res.solution_errors)
            med_res.not_solve_rate = med_res.solution_errors.count(10) / len(med_res.solution_errors)
        # 每个矿工的工作量
        for miner_id in med_res.miner_chains.keys():
            med_res.miner_chains[miner_id] = med_res.miner_chains[miner_id] / len(med_res.solve_rounds)
            med_res.miner_mains[miner_id] = med_res.miner_mains[miner_id] / len(med_res.solve_rounds)
            med_res.miner_unpubs[miner_id] = med_res.miner_unpubs[miner_id] / len(med_res.solve_rounds)
    
    file_name = f'intermediate_m{m}d{d}v{v}.json'
    with open(result_path / file_name, 'w+') as f:
        sdspec_dict = asdict(med_res)
        filtered_dict = {
            key: value
            for key, value in sdspec_dict.items()
            if not (
                (isinstance(value, (int, float)) and value == 0) or
                (isinstance(value, (list, dict)) and not value)
            )
        }

        json_res = json.dumps(filtered_dict)
        f.write(json_res)
    sd_lite = med_res_to_lite_res(med_res)
    res_collect.data_list.append(sd_lite)
    # 缓存仿真结果
    cache_name = f"collect_before_m{m}d{d}v{v}.json"
    simudata_collect_to_json(res_collect, cache_path, cache_name)



"""keyblock分叉率仿真，与前面不同的是该仿真为一长链"""
@dataclass
class simudata_kbfr:
    var_num:str
    difficulty:int
    miner_num:int
    kb_strategy:str
    safe_thre:float
    kb_forkrates:list[float]
    kb_nums:list[int]
    kb_forknums:list[int]
    mb_forkrates:list[float]
    mb_nums:list[int]
    accept_mb_nums:list[int]
    ave_kb_forkrate:float
    total_kb_forkrate:float

def simudata_kbfr_to_json(sd_kbfr: simudata_kbfr, file_path, file_name=None):
    if file_name is None:
        file_name = "simudata_kbfr.json"
    with open(file_path / file_name, "w+") as f:
        sd_kbfr_dict = asdict(sd_kbfr)
        kbfr_json = json.dumps(sd_kbfr_dict)
        f.write(kbfr_json)

def simu_mbkb_forkrate_longchain(
        context:Background, iter_num:int, kb_strategy:str, 
        safe_thre:float, difficulty:int, var_num:int, 
        miner_num:int, prblm_pool_method, pool_size
    ):
    RESULT_PATH = (context.get_result_path() / 
                    f"kbfr_strategy{kb_strategy}d{difficulty}-{mp.current_process().name}")
    RESULT_PATH.mkdir()
    CACHE_PATH = RESULT_PATH / 'kbfr_cache'
    CACHE_PATH.mkdir()
    context.set_keyblock_strategy(kb_strategy)
    print(f"Loading problem pool--{mp.current_process().name}")
    prblm_pool = get_prblm_pool(pool_size, var_num, prblm_pool_method, RESULT_PATH)
    print(f"Finish loading--{mp.current_process().name}")
    sd_kbfr = simudata_kbfr(var_num, difficulty, miner_num, kb_strategy, safe_thre, [], [], [], [], [],[], 0, 0)
    for i in  range(iter_num):
        Env = new_environment(context, prblm_pool[0], miner_num, difficulty)
        Env.env_load_prblm_pool(prblm_pool[1:pool_size])
        Env.context.set_test_prblm(prblm_pool[0])
        Env.exec()
        eva_res = Env.view()
        record_intermediate_kbfr(sd_kbfr,eva_res)
        simudata_kbfr_to_json(sd_kbfr, CACHE_PATH, f'kbfr_cache_iter{iter_num}.json')
    # 结果保存
    simudata_kbfr_to_json(sd_kbfr, RESULT_PATH, f'kbfr_cache_d{difficulty}_{kb_strategy}.json')
    
        
def record_intermediate_kbfr(simudata_kbfr:simudata_kbfr, eva_res:EvaResult):
    simudata_kbfr.kb_forkrates.append(eva_res.kb_forkrate)
    simudata_kbfr.kb_nums.append(eva_res.kb_num)
    simudata_kbfr.kb_forknums.append(eva_res.kb_forknum)
    simudata_kbfr.ave_kb_forkrate = sum(simudata_kbfr.kb_forkrates)/len(simudata_kbfr.kb_forkrates)
    simudata_kbfr.total_kb_forkrate = sum(simudata_kbfr.kb_forknums)/sum(simudata_kbfr.kb_nums)
