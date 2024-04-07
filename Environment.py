import copy
import gc
import json
import logging
import math
import multiprocessing as mp
import random
import time

import numpy as np

import network
from background import Background
from chain import Block, Chain, NewBlocks
from evaluation import Evaluation
from functions import for_name
from miner import Miner
from myattack import default_attack_mode


def get_time(f):
    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

logger = logging.getLogger(__name__)
class Environment(object):
    def __init__(self,  
            background:Background, 
            t:int = None, 
            q_ave:int = None, 
            q_distr:str = None, 
            target:str = None, 
            adversary_ids:tuple = None, 
            network_param:dict = None, 
            genesis_prblm = None,
            recordSols:bool = False):
        self.background = background
        self.evaluation = Evaluation(self.background, recordSols)
        #environment parameters
        self.miner_num = self.background.get_miner_num()  # number of miners
        self.max_adversary = t  # maximum number of adversary
        self.q_ave = q_ave  # number of hash trials in a round
        self.q_distr = [] #
        self.target = target
        self.total_round = 0
        self.global_chain = Chain(self.background)
        # generate miners
        self.miners:list[Miner] = []
        if q_distr == 'rand':
            self.create_miners_q_rand()
        elif q_distr == 'equal':
            self.create_miners_q_equal()
        self.env_create_genesis_block(genesis_prblm)
        # generate network
        self.network:network.Network = \
            for_name(self.background.get_network_type())(self.background, self.miners)
        self.network.set_net_param(**network_param)
        # 初始化攻击模组
        if t >0:
            self.init_attack_mode(t,adversary_ids)
        # evaluation
        self.max_suffix = 10
        # 每轮结束时，各个矿工的链与common prefix相差区块个数的分布
        self.cp_pdf = np.zeros((1, self.max_suffix)) 
        self.view_miner:Miner = None
           
        
    def init_attack_mode(self, t, adversary_ids):
        self.max_adversary = t  # maximum number of adversary
        self.adversary_mem:list[Miner] = []
        if adversary_ids is not None:
            if len(adversary_ids) != self.max_adversary:
                self.max_adversary = len(adversary_ids)
            self.select_adversary(*adversary_ids)
        elif self.max_adversary >= 0:
            self.select_adversary_random()
            adversary_ids = [adversary.miner_id for adversary in self.adversary_mem]
        else:
            raise ValueError("Wrong attacker selection")
        if self.adversary_mem: # 如果有攻击者，则创建攻击实例
            self.attack = default_attack_mode(self.background, self.evaluation, 
                self.adversary_mem, self.global_chain, self.network)
            self.ad_main = random.choice(self.adversary_mem)
        self.attack_execute_type = self.background.get_attack_execute_type()

    def select_adversary_random(self):
        '''
        随机选择对手
        return:self.adversary_mem
        '''
        miners = [m for m in self.miners if m.miner_id != 0]
        self.adversary_mem=random.sample(miners, self.max_adversary)
        for adversary in self.adversary_mem:
            adversary.set_adversary(True)
        return self.adversary_mem


    def select_adversary(self,*Miner_ID):
        '''
        根据指定ID选择对手
        return:self.adversary_mem
        '''   
        for miner in Miner_ID:
            self.adversary_mem.append(self.miners[miner])
            self.miners[miner].set_adversary(True)
        return self.adversary_mem
    
    def clear_adversary(self):
        '''清空所有对手'''
        for adversary in self.adversary_mem:
            adversary.set_adversary(False)
        self.adversary_mem=[]

    def create_miners_q_equal(self):
        for miner_id in range(self.miner_num):
            self.miners.append(Miner(self.background, miner_id, 
                    self.q_ave, self.target, self.evaluation))

    def create_miners_q_rand(self):
        '''
        随机设置各个节点的hash rate,满足均值为q_ave,方差为1的高斯分布
        且满足全网总算力为q_ave*miner_num
        '''
        # 生成均值为ave_q，方差为1的高斯分布
        q_dist = np.random.normal(self.q_ave, 0.2*self.q_ave, self.miner_num)

        # 归一化到总和为total_q，并四舍五入为整数
        total_q = self.q_ave * self.miner_num
        q_dist = total_q / np.sum(q_dist) * q_dist
        q_dist = np.round(q_dist).astype(int)
        # 修正，如果和不为total_q就把差值分摊在最小值或最大值上
        if np.sum(q_dist) != total_q:
            diff = total_q - np.sum(q_dist)
            for _ in range(abs(diff)):
                sign_diff = np.sign(diff)
                idx = np.argmin(q_dist) if sign_diff > 0 else np.argmax(q_dist)
                q_dist[idx] += sign_diff
        for miner_id,q in zip(range(self.miner_num), q_dist):
            self.miners.append( Miner(self.background, miner_id, q, 
                                    self.target, self.evaluation))
        return q_dist
    
    def env_load_prblm_pool(self, prblm_pool:list):
        for miner in self.miners:
            miner.txpool.load_prblm_pool(prblm_pool)

    def env_create_genesis_block(self, o_prblm):
        '''create genesis block for all the miners in the system.'''
        origin_prblm = self.background.get_genesis_prblm()
        # origin_prblm = prblm_generator()
        # origin_prblm = o_prblm
        
        self.global_chain.create_genesis_block(copy.deepcopy(origin_prblm))
        for miner in self.miners:
            miner.local_chain.create_genesis_block(copy.deepcopy(origin_prblm))
            miner.consensus.cur_keyblock = miner.local_chain.head
            miner.consensus.open_blocks.append(miner.local_chain.head)
            miner.consensus.cur_keyid = 0

    def add_miniblock_to_global_chain(self, new_mb:Block):
        """
        将miniblock复制后添加到全局链上
        """
        copyblk = copy.deepcopy(new_mb)
        block_to_link = \
            self.global_chain.search_forward_by_hash(new_mb.blockhead.prehash)
        self.global_chain.\
            add_block_direct(copyblk, block_to_link)
        # 与该区块branch的子问题建立连接
        for prblm in block_to_link:
            if prblm.pname == new_mb.minifield.pre_pname:
                copyblk.minifield.pre_p = prblm
        # 更新状态树
        copyblk.minifield.update_fathomed_state()
        if copyblk.minifield.bfthmd_state:
            copyblk.update_solve_tree_fthmd_state()

    def add_global_chain(self, newblock:NewBlocks):
        """
        将新产生的区块加入到全局链
        """
        if not newblock.iskeyblock:
            new_mb = newblock.miniblock
            self.add_miniblock_to_global_chain(new_mb)
            return
        new_kb = newblock.keyblock
        newmbs_unsafe = newblock.mbs_unsafe
        for mb_undafe in newmbs_unsafe:
            self.add_miniblock_to_global_chain(mb_undafe)
        copy_kb = copy.deepcopy(new_kb)
        block_to_link = \
            self.global_chain.search_forward_by_hash(new_kb.blockhead.prehash)
        self.global_chain.\
            add_block_direct(copy_kb, block_to_link)
        pre_kb = new_kb.keyfield.pre_kb
        # 如果没有pre_keyblock，不需要建立连接
        if pre_kb is None:
            return
        # 建立keyblock间的连接
        global_kbs = self.global_chain.get_keyblocks_pref()
        for kb in global_kbs:
            if pre_kb.blockhead.blockhash != kb.blockhead.blockhash:
                continue
            copy_kb.keyfield.pre_kb = kb
            kb.keyfield.next_kbs.append(copy_kb)
            logger.info(f"global add keyblock "
                        f"{copy_kb.name} success, pre_keyblock{copy_kb.keyfield.pre_kb.name}, "
                        f"next keyblocks of {copy_kb.keyfield.pre_kb.name}: "
                        f"{[b.name for b in kb.keyfield.next_kbs]}")
        if self.background.get_keyblock_strategy() != 'pow':
            mb_with_kb = newblock.mb_with_kb
            self.add_miniblock_to_global_chain(mb_with_kb)

    def attack_execute(self,round):
        return self.attack.mine_steal(round)
        # if self.attack_execute_type == 'execute_sample0':
        #     self.attack.execute_sample0(round)
        # elif self.attack_execute_type == 'execute_sample1':
        #     self.attack.execute_sample1(round)
        # else:
        #     print('Undefined attack mode, please check the system_config.ini')
        #     exit()
    
    #@get_time
    def exec(self, evn_exec_done = None, terminate_event = None, ERROR_PATH = None, quiet = True):
        '''
        调用当前miner的BackboneProtocol完成mining
        当前miner用addblock功能添加上链
        之后gobal_chain用深拷贝的addchain上链
        '''
        # logger.info('*'*20 + f"{self.miner_num}Miner(s)  "+ 'New exec!' + '*'*20)
        max_rounds = self.background.get_total_round()
        # self.view_miner = self.miners[0]
        t_0 = time.time()
        t_gc = t_0
        for round in range(1, max_rounds+1): 
            for miner in self.miners:
                if terminate_event and terminate_event.is_set():
                    if evn_exec_done:
                        evn_exec_done.set()
                    print(f"Env terminate!--{mp.current_process().name}")
                    return
                if miner.isAdversary and miner.miner_id != self.ad_main.miner_id:
                    continue
                if miner.isAdversary and miner.miner_id == self.ad_main.miner_id:
                    logger.info(f"Miner{miner.miner_id}: start attack")
                    stealed_newmbs = self.attack_execute(round)
                    # if len(stealed_newmbs) > 0:
                    #     for newmb in stealed_newmbs:
                    #         self.add_global_chain(newmb)
                    for ad in self.adversary_mem:
                        ad.input_tape = []  # clear the input tape
                        ad.receive_tape = []  # clear the communication tape
                    continue
                # 执行挖矿程序
                newblock = miner.backbone_protocol(round)
                miner.input_tape = []  # clear the input tape
                miner.receive_tape = []  # clear the communication tape
                # # 监测动态解
                if self.evaluation.recordSols and (newblock is None or 
                    (newblock is not None and not newblock.iskeyblock)):
                    # if miner.miner_id == self.view_miner.miner_id:
                    # self.evaluation.record_relaxed_solutions(miner, round)
                    self.evaluation.record_lowerbound(miner, round)
                if newblock is None:
                    continue
                # 新挖出的矿进入网络
                self.network.access_network(newblock, miner.miner_id, round)
                # 接入全局链
                # self.add_global_chain(newblock)
                if not newblock.iskeyblock:
                    continue
                # 如果是keyblock，重设问题编号
                # self.global_var.reset_prblm_number()
                # 问题池中无更多问题，结束运行
                if newblock.keyblock.keyfield.key_tx is None:
                    self.network.diffuse(round)
                    for other_miner in self.miners:
                        other_miner.backbone_protocol(round)
                    self.view_miner = miner
                    self.total_round = round
                    if self.evaluation.recordSols:
                        self.evaluation.get_ubs(self.view_miner.local_chain)
                    self.background.reset_id_center()
                    if (len(self.miners[0].local_chain.get_feasible_keyblocks())==0 
                        and ERROR_PATH is not None):
                        self.save_err_prblm(ERROR_PATH, "a_infeasible_prblms.json")
                    if evn_exec_done:
                        evn_exec_done.set()
                    return round
            self.network.diffuse(round)  # diffuse(C)
            if quiet is True:
                continue
            self.process_bar(round, max_rounds, t_0) # 进度条
            # if time.time()-t_gc >= 300:
            #     print("garbage collecting...",end=" ")
            #     gc.collect()
            #     time.sleep(10)
            #     t_gc = time.time()
        self.view_miner = self.miners[0]
        self.total_round = max_rounds
        self.background.reset_id_center()
        if ERROR_PATH is not None:
            self.save_err_prblm(ERROR_PATH, "a_hard_prblms.json")
        if evn_exec_done:
            evn_exec_done.set()
        return  self.total_round
    
    def save_err_prblm(self, ERROR_PATH, file_name):
        with open(ERROR_PATH / file_name, "a+") as f:
            prblm = self.background.get_genesis_prblm()
            p_json = json.dumps({
                'c': (-prblm.c).tolist(),
                'G_ub':prblm.G_ub.tolist(),
                'h_ub':prblm.h_ub.tolist()})
            f.write(p_json + '\n')



    def view(self, quiet = True, pool_path=None):
        # 展示一些仿真结果
        # print('\n')
        # # for miner_i in range(self.miner_num):
        # #     self.miners[miner_i].local_chain.printchain2txt(f"chain_data{str(miner_i)}.txt")
        # self.global_chain.printchain2txt("global_chain_data.txt")
        # print("Global Tree Structure:", "")
        # self.global_chain.ShowStructure1()
        # print("End of Global Tree", "")
        # # Evaluation Results
        if self.view_miner is None:
            self.view_miner = self.miners[0]
        self.evaluation.cal_packaged_results(self.view_miner.local_chain)
        evaluation_result = self.evaluation.collect_evaluation_results()
        if not quiet:
            print(f"view miner: {self.view_miner.miner_id}")
            self.evaluation.save_results_to_json(pool_path)
            # self.global_chain.printchain2txt()
            self.view_miner.local_chain.printchain2txt()
            # self.global_chain.ShowStructureWithGraphviz()
            self.view_miner.local_chain.show_chain_by_graphviz()
            # self.attack.save_draw_final_success_rates()
        return evaluation_result



    def process_bar(self,process,total,t_0):
        bar_len = 50
        percent = (process)/total
        cplt = "■" * math.ceil(percent*bar_len)
        uncplt = "□" * (bar_len - math.ceil(percent*bar_len))
        time_len = time.time()-t_0+0.0000000001
        time_cost = time.gmtime(time_len)
        vel = process/(time_len)
        time_eval = time.gmtime(total/(vel+0.001))
        print("\r{}{}  {:.5f}%  {}/{}  {:.2f} round/s  {}:{}:{}>>{}:{}:{} "\
        .format(cplt, uncplt, percent*100, process, total, vel, time_cost.tm_hour, time_cost.tm_min, time_cost.tm_sec,\
            time_eval.tm_hour, time_eval.tm_min, time_eval.tm_sec),end="", flush=True)
