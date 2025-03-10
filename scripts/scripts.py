import json
import os
import queue
import shutil
import threading
import time
import traceback as tc
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pulp
from pysat.solvers import Solver
from scipy.optimize import linprog


def copy_files_from_txt(txt_file_path, source_folder, target_folder):
    """
    从包含文件名的.txt文件中找到匹配的文件并复制到目标文件夹。

    Parameters:
    - txt_file_path (str): 包含文件名的.txt文件的路径。
    - source_folder (str): 包含所有文件的源文件夹的路径。
    - target_folder (str): 目标文件夹的路径。
    """
    # 创建目标文件夹
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 读取.txt文件中的文件名
    with open(txt_file_path, 'r') as txt_file:
        file_names = txt_file.read().splitlines()

    # 遍历源文件夹中的文件
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file in file_names:
                # 构建源文件路径和目标文件路径
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_folder, file)

                # 复制文件到目标文件夹
                shutil.copy2(source_file_path, target_file_path)

    print("文件复制完成")



def monitor_memory_threshold(
        process:psutil.Process,
        get_var_thread_done:threading.Event,
        continue_flag:threading.Event,
    ):
    """
    内存实时监测模块
    """
    # memory_threshold = 1556480
    # print("monitoring start")
    memory_threshold = 1.5 * 1024 * 1024 * 1024
    while not get_var_thread_done.is_set():
        # 检查当前进程的内存占用
        # print("monitoring")
        current_memory_usage = process.memory_info().rss
        if current_memory_usage > memory_threshold:
            # 内存占用超过阈值，保存参数信息到txt文件
            continue_flag.set()  # 通知主线程继续循环
            print("Memory usage exceeded. Skipping...")
            return# 退出内存监测循环
        else:
            # 内存占用未超过阈值，继续监测
            time.sleep(1)  # 每秒检查一次，可以根据需要调整时间间隔
    # get_var_thread_done.clear()
    # print("monitoring finished")

def get_var_num_from_mps(file_path, q:queue.Queue, get_var_thread_done:threading.Event):
    """中提取变量的数量"""
    try:
        var,lp = pulp.LpProblem.fromMPS(file_path)
        integer_vars = [var for var in lp.variables() if var.cat == pulp.LpInteger]
        continuous_vars = [var for var in lp.variables() if var.cat == pulp.LpContinuous]
        len(lp.constraints)
        q.put((len(var),len(integer_vars), len(continuous_vars), len(lp.constraints)))
        return len(var),len(integer_vars), len(continuous_vars), len(lp.constraints)
    except Exception:
        tc.print_exc()
        # get_var_thread_done.set()
        q.put((0,0,0,0))
        return (0,0,0,0)

def find_easy_mps(source_path):
    """在指定文件夹中找到小于目标var_num的MPS文件"""
    current_process_id = os.getpid()
    current_process = psutil.Process(current_process_id)
    # 初始化
    var_num_thred = 1000
    target_folder = 'E:\Files\A-blockchain\\branchbound\MIPLIB2017\easiest_1000_2'
    var_counts = []
    intvar_counts = []
    contvar_counts = []
    cons_counts = []
    mps_files = []
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    ## mod011
    for root, dirs, files in os.walk(source_path):
        for i, mps_file in enumerate(files):
            print(i, mps_file)
            continue_flag = threading.Event()
            get_var_thread_done = threading.Event()
            # 启动get_var线程
            q = queue.Queue()
            get_var_thread = threading.Thread(
                target=get_var_num_from_mps,
                args=(os.path.join(root, mps_file),q, get_var_thread_done))
            get_var_thread.start()
            # 启动内存监测线程
            memory_monitor_thread = threading.Thread(
                target=monitor_memory_threshold,
                args=(current_process, get_var_thread_done, continue_flag, ))
            memory_monitor_thread.start()
            # 等待get_var线程完成
            get_var_thread.join()
            var_num, intvar, contvar, cons = q.get()
            print(f"File: {mps_file}, c length: {var_num}")
            var_counts.append(var_num)
            intvar_counts.append(intvar)
            contvar_counts.append(contvar)
            cons_counts.append(cons)
            mps_files.append(mps_file)
            if var_num < var_num_thred:
                # 构建源文件路径和目标文件路径
                source_file_path = os.path.join(root, mps_file)
                target_file_path = os.path.join(target_folder, mps_file)

                # 复制文件到目标文件夹
                shutil.copy2(source_file_path, target_file_path)
            # 通知内存监测线程get_var线程已完成
            get_var_thread_done.set()
            # 如果监测超出内存限制，执行continue
            if continue_flag.is_set() or get_var_thread_done.is_set():
                if continue_flag.is_set():
                    continue_flag.clear()
                if get_var_thread_done.is_set():
                    get_var_thread_done.clear()
                continue
    print("finished!")
    # 创建数据框
    data = {
        'MPS File': mps_files,
        'Variable Count': var_counts,
        'Integer Variable Count': intvar_counts,
        'Non-Integer Variable Count': contvar_counts,
        'Constraint Count': cons_counts
    }
    df = pd.DataFrame(data)

    # 保存数据框到CSV文件
    # df.to_csv("mps.csv", index=False)
    df.to_excel('output3.xlsx', index=False)
    print("Saved results to mps.csv")

def merge_jsons(folder_path, output_file,file_cate):
    """
    将文件夹下所有名为simudata_collect的json文件合并
    """
    for root, dirs, files in os.walk(folder_path):
        with open(output_file, 'a', encoding='utf-8') as output:
            for filename in files:
                 if file_cate in filename and filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    # 逐行读取JSON文件的内容并逐行写入总的JSON文件
                    with open(file_path, 'r', encoding='utf-8') as file:
                        for line in file:
                            output.write(line.strip() + '\n')

def delete_atklog_fromjson(file_path, out_path):
    with open(file_path, 'r') as f:
        ret_json_list = f.read().split('\n')[:-1]
        for ret_json in ret_json_list:
            ret = json.loads(ret_json)
            new_ret = {}
            for (k,v) in zip(ret.keys(),ret.values()):
                if k in ["atklog_depth", "atklog_mb"]:
                    continue
                new_ret.update({k:v})
            with open(out_path, "a") as f1:
                new_json = json.dumps(new_ret)
                f1.write(new_json + '\n')

def merge_times(folder_path, out_path):
    new_times = {
        "mb_times" : [],
        "unpub_times" : [],
        "fork_times" : [],
        "kb_times": []
    }
    new_res = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not('simudata_collect' in file and file.endswith('.json')):
                continue
            with open(os.path.join(root, file), 'r') as f:
                ret_json_list = f.read().split('\n')[:-1]
                for ret_json in ret_json_list:
                    ret = json.loads(ret_json)
                    for (k,v) in zip(ret.keys(),ret.values()):
                        new_res.update({k:v})
                        if k in new_times.keys():
                            new_times[k].extend(v)
    new_res.update(new_times)
    with open(out_path, "w") as f1:
        new_json = json.dumps(new_res)
        f1.write(new_json + '\n')

def merge_intermediate_data(folder_path, out_path):
    def cal_average(list_data:list):
        return sum(list_data)/len(list_data)
    new_res = {
        'subpair_nums': [],
        'subpair_unpubs': [],
        'mb_nums':[],
        'accept_mb_nums':[],
        'mb_forkrates':[],
    }
    ave_res = {
        'subpair_num':0,
        'subpair_unpub':0,
        'mb_forkrate':0,
        'ave_mbforrate':0,
    }
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if not('intermediate' in filename and filename.endswith('.json')):
                continue
            with open(os.path.join(root, filename), 'r') as f:
                # res_json_list = f.read().split('\n')[:-1]
                # for res_json in res_json_list:
                res = json.load(f)
                for (k,v) in zip(res.keys(),res.values()):
                    if k in new_res.keys():
                        new_res[k].extend(v)
    ave_res['subpair_num'] = cal_average(new_res['subpair_nums'])
    ave_res['subpair_unpub'] = cal_average(new_res['subpair_unpubs'])
    ave_res['mb_forkrate'] = (sum(new_res['mb_nums'])-sum(new_res['accept_mb_nums']))/sum(new_res['mb_nums'])
    ave_res['ave_mbforrate'] = cal_average(new_res['mb_forkrates'])
    ave_res.update(new_res)
    with open(out_path, "w") as f1:
        new_json = json.dumps(ave_res)
        f1.write(new_json + '\n')

def merge_evares_jsons(folder, output_file):
    for root, dirs, files in os.walk(folder):
        with open(output_file, 'a', encoding='utf-8') as output:
            for filename in files:
                 if 'evaluation' in filename and filename.endswith('.json'):
                    file_path = os.path.join(root, filename)
                    # 逐行读取JSON文件的内容并逐行写入总的JSON文件
                    with open(file_path, 'r', encoding='utf-8') as file:
                        for line in file:
                            output.write(line)
                    output.write('\n')


def cult_tsp_solving_process(file_path, out_path):
    out_data = {}
    with open(file_path, 'r') as f:
        data = json.load(f)
        for k, v in data.items():
            if k == "lowerbounds":
                continue
            if k != "ubdata":
                out_data.update({k:v})
                continue
            
            ubs = []
            for i, ub in enumerate(v):
                if i >= 5000:
                    break
                ubs.append(ub)
            out_data.update({k:ubs})
                
    with open(out_path, 'w') as fo:
        jd = json.dumps(out_data)
        fo.write(jd)

def remove_duplicates(main_file_path: str, reference_file_path: str, output_file_path: str):
    """
    从main_file中删除在reference_file中出现过的条目（只比较第一个字段）
    
    Args:
        main_file_path: 需要处理的主文件路径
        reference_file_path: 参考文件路径，用于检查重复
        output_file_path: 输出文件路径
    
    Returns:
        tuple: (总行数, 删除的行数)
    """
    # 读取参考文件中的所有条目的第一个字段
    reference_first_fields = set()
    with open(reference_file_path, 'r') as f:
        for line in f:
            if line.strip():  # 忽略空行
                try:
                    json_obj = json.loads(line.strip())
                    # 获取第一个字段的值
                    first_field = next(iter(json_obj.values()))
                    reference_first_fields.add(str(first_field))
                except (json.JSONDecodeError, StopIteration):
                    continue
    
    # 处理主文件，只保留不重复的条目
    total_lines = 0
    kept_lines = 0
    with open(main_file_path, 'r') as f_in, open(output_file_path, 'w') as f_out:
        for line in f_in:
            if line.strip():  # 只统计非空行
                total_lines += 1
                try:
                    json_obj = json.loads(line.strip())
                    # 获取第一个字段的值
                    first_field = next(iter(json_obj.values()))
                    if str(first_field) not in reference_first_fields:
                        f_out.write(line)
                        kept_lines += 1
                except (json.JSONDecodeError, StopIteration):
                    # 如果解析失败，保留该行
                    f_out.write(line)
                    kept_lines += 1

    removed_lines = total_lines - kept_lines
    print(f"处理完成。输出文件已保存到: {output_file_path}")
    print(f"总行数: {total_lines}")
    print(f"删除行数: {removed_lines}")
    print(f"保留行数: {kept_lines}")
    
    return total_lines, removed_lines




if __name__ == "__main__":
    import random

    # folder_path = ".\Results\\20231203\\230820"
    # output_path = ".\Results\\20231203\\230820\\times12prob0_3m3.json"
    # med_path =    ".\Results\\20231203\\230820\\res12prob0_3m3.json"
    # merge_times(folder_path, output_path)
    # merge_intermediate_data(folder_path, med_path)
    folder_path = "E:\Files\gitspace\\bbb-github\Results\\20250309\\230617"
    output_path = "E:\Files\gitspace\\bbb-github\Results\\20250309\\230617\\final_results.json"
    merge_jsons(folder_path, output_path, "final_results")
    # merge_jsons(folder_path, output_path, "intermediate")
    # load_tsp()
    # for root, dirs, files in os.walk(f"E:\Files\A-blockchain\\branchbound\MAXSAT\EASY"):
    #     for file in files:
    #         load_maxsat(file)
    # cult_tsp_solving_process("Result_Data\\tsp solving process.json", "Result_Data\\tsp solving process2.json")
    # merge_evares_jsons(folder_path, output_path)

    # 使用示例
    # main_file = "E:\Files\gitspace\\bbb-github\Problem Pools\SPOT\Generated2\\30_1.json"  # 替换为您的主文件路径
    # reference_file = "E:\Files\gitspace\\bbb-github\Results\\20250305\\235103\short_g5000v30m[5]d[5]--003356-44616\errorlogs_v30d5m5a0\\a_bad_prblms.json"  # 替换为您的参考文件路径
    # output_file = "E:\Files\gitspace\\bbb-github\Problem Pools\SPOT\Generated2\\30_1_filtered.json"  # 替换为您想要的输出文件路径
    
    # remove_duplicates(main_file, reference_file, output_file)