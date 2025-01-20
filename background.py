'''
    管理上下文信息
'''
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
import data.lpprblm

RESULT_PATH=Path.cwd() / 'Results' / time.strftime("%Y%m%d") / time.strftime("%H%M%S")


class Background(object):
    def __init__(self):
        '''
        初始化
        '''
        if not os.path.exists(RESULT_PATH):
            print(f"loading background with ROOT PATH: {RESULT_PATH}")
            RESULT_PATH.mkdir(parents=True, exist_ok=True)
        NET_RESULT_PATH=RESULT_PATH / 'Network Results'
        if not os.path.exists(NET_RESULT_PATH):
            NET_RESULT_PATH.mkdir(parents=True, exist_ok=True)
        CHAIN_DATA_PATH=RESULT_PATH / 'Chain Data'
        if not os.path.exists(CHAIN_DATA_PATH):
            CHAIN_DATA_PATH.mkdir(parents=True, exist_ok=True)
        self._var_dict = {}
        self._var_dict['MINER_NUM']=0
        self._var_dict['POW_TARFET']=''
        self._var_dict['AVE_Q']=0
        self._var_dict['TOTAL_ROUND']=0
        self._var_dict['CONSENSUS_TYPE']='consensus.PoW'
        self._var_dict['NETWORK_TYPE']='network.FullConnectedNetwork'
        self._var_dict['BLOCK_NUMBER'] = 0
        self._var_dict['PRBLM_NUMBER'] = 0
        self._var_dict['KEY_PRBLM_NUMBER'] = 0
        self._var_dict['RESULT_PATH'] = RESULT_PATH
        self._var_dict['NET_RESULT_PATH'] = NET_RESULT_PATH
        self._var_dict['CHAIN_DATA_PATH'] = CHAIN_DATA_PATH
        self._var_dict['Attack'] = False
        self._var_dict['Blocksize'] = 2
        self._var_dict['LOG_LEVEL'] = logging.ERROR
        self._var_dict['Show_Fig'] = False
        self._var_dict['BB_Diffculty'] = 1
        self._var_dict['Genesis_Prblm'] = None
        self._var_dict['SAFE_THRE'] = 1
        self._var_dict["VAR_NUM"] = 20
        self._var_dict['KEY_BLOCK_ST'] = 'pow'
        self._var_dict['ATTACK_EXECUTE_TYPE']='execute_sample0'
        self._var_dict['SOLVE_PROB'] = 0.5
        self._var_dict['OPEN_BLOCK_ST'] = "openblock_random"
        self._var_dict['OPEN_PROBLEM_ST'] = "openprblm_random"
        self._var_dict['total_gas'] = 10000
        self._var_dict['once_gas'] = 1
        # 管理问题自增id
        self._prblm_id_center = defaultdict(int)

        self._gases = defaultdict(int)
    
    # 路径管理
    def get_result_path(self)->Path:
        return self._var_dict['RESULT_PATH']
    def set_result_path(self, result_path):
        self._var_dict['RESULT_PATH'] = result_path
        if not os.path.exists(result_path):
            print(f"loading background with ROOT PATH: {result_path}")
            result_path.mkdir(parents=True, exist_ok=True)
        self.set_net_result_path(result_path / 'Network Results')
        self.set_chain_data_path(result_path / 'Chain Data')

    def get_net_result_path(self):
        return self._var_dict['NET_RESULT_PATH']
    def set_net_result_path(self, net_result_path):
        self._var_dict['NET_RESULT_PATH']= net_result_path
        if not os.path.exists(net_result_path):
            print(f"loading background with NET PATH: {net_result_path}")
            net_result_path.mkdir(parents=True, exist_ok=True)

    def get_chain_data_path(self):
        return self._var_dict['CHAIN_DATA_PATH']
    def set_chain_data_path(self,chain_result_path):
        self._var_dict['CHAIN_RESULT_PATH']= chain_result_path
        if not os.path.exists(chain_result_path):
            print(f"loading background with CHAIN DATA PATH: {chain_result_path}")
            chain_result_path.mkdir(parents=True, exist_ok=True)
    # 日志
    def set_log_level(self, log_level):
        '''设置日志级别'''
        self._var_dict['LOG_LEVEL'] = log_level
    def get_log_level(self):
        '''获得日志级别'''
        return self._var_dict['LOG_LEVEL']
    # 共识
    def set_consensus_type(self, consensus_type):
        '''定义共识协议类型 type:str'''
        self._var_dict['CONSENSUS_TYPE'] = consensus_type
    def get_consensus_type(self):
        '''获得共识协议类型'''
        return self._var_dict['CONSENSUS_TYPE']
    # 全局不变量
    def set_miner_num(self,miner_num):
        '''定义矿工数量 type:int'''
        self._var_dict['MINER_NUM'] = miner_num
    def get_miner_num(self):
        '''获得矿工数量'''
        return self._var_dict['MINER_NUM']
    def set_total_round(self,total_round):
        self._var_dict['TOTAL_ROUND']=total_round
    def get_total_round(self):
        return self._var_dict['TOTAL_ROUND']
    def set_PoW_target(self,PoW_target):
        '''定义pow目标 type:str'''
        self._var_dict['POW_TARFET'] = PoW_target
    def get_PoW_target(self):
        '''获得pow目标'''
        return self._var_dict['POW_TARFET']
    def set_ave_q(self,ave_q):
        '''定义pow,每round最多hash计算次数 type:int'''
        self._var_dict['AVE_Q'] = ave_q
    def get_ave_q(self):
        '''获得pow,每round最多hash计算次数'''
        return self._var_dict['AVE_Q']
    def set_blocksize(self,blocksize):
        self._var_dict['Blocksize'] = blocksize
    def get_blocksize(self):
        return self._var_dict['Blocksize']

    def set_network_type(self,network_type):
        '''定义网络类型 type:str'''
        self._var_dict['NETWORK_TYPE'] = network_type
    def get_network_type(self):
        '''获得网络类型'''
        return self._var_dict['NETWORK_TYPE']
    # 一些开关
    def activate_attacktion(self):
        self._var_dict['Attack'] = True
    def set_show_fig(self,show_fig):
        self._var_dict['Show_Fig'] = show_fig
    def get_show_fig(self):
        return self._var_dict['Show_Fig']

    def save_configuration(self):
        '''将self._var_dict中的内容保存到configuration.txt中'''
        with open(self._var_dict['RESULT_PATH'] / "configuration.txt",
                'w+') as config:
            for key,value in self._var_dict.items():
                print(key,": ",value,file=config)
    # 攻击
    def set_attack_execute_type(self,attack_type):
        self._var_dict['ATTACK_EXECUTE_TYPE'] =  attack_type
    def get_attack_execute_type(self):
        return self._var_dict['ATTACK_EXECUTE_TYPE'] 
    # 全局自增区块id
    def get_block_number(self):
        '''获得产生区块的独立编号'''
        self._var_dict['BLOCK_NUMBER'] = self._var_dict['BLOCK_NUMBER'] + 1
        return self._var_dict['BLOCK_NUMBER']
    def reset_block_number(self):
        self._var_dict['BLOCK_NUMBER'] = 0

    # branchbound 
    # key-problem 编号
    def get_key_prblm_number(self):
        self._var_dict['KEY_PRBLM_NUMBER'] = self._var_dict['KEY_PRBLM_NUMBER'] + 1
        return self._var_dict['KEY_PRBLM_NUMBER']
    def reset_key_prblm_number(self):
        self._var_dict['KEY_PRBLM_NUMBER'] = 0
    # 获取子问题编号
    def get_prblm_number(self):
        self._var_dict['PRBLM_NUMBER'] = self._var_dict['PRBLM_NUMBER'] + 1
        return self._var_dict['PRBLM_NUMBER']
    def reset_prblm_number(self):
        self._var_dict['PRBLM_NUMBER'] = 0
    # 单轮求解概率
    def set_solve_prob(self, solve_prob:float):
        self._var_dict['SOLVE_PROB'] = solve_prob
    def get_solve_prob(self):
        return self._var_dict['SOLVE_PROB']
    # 设置创世块问题
    def set_genesis_prblm(self, test_prblm):
        self._var_dict['Genesis_Prblm'] = test_prblm
    def get_genesis_prblm(self):
        return self._var_dict['Genesis_Prblm']
    # miniblock求解难度
    def set_bb_difficulty(self,bb_difficulty):
        self._var_dict['BB_Difficulty'] = bb_difficulty
    def get_bb_difficulty(self):
        return self._var_dict['BB_Difficulty']
    # 安全性阈值
    def set_safe_thre(self,safe_thre):
        self._var_dict['SAFE_THRE'] = safe_thre
    def get_safe_thre(self):
        return self._var_dict['SAFE_THRE']
    # 变量数量(问题规模)
    def set_var_num(self,var_num):
        self._var_dict['VAR_NUM'] = var_num
    def get_var_num(self):
        return self._var_dict['VAR_NUM']
    # 产生keyblock的策略：'pow'/'withmini'/'pow+withmini'
    def set_keyblock_strategy(self, keyblock_strategy):
        self._var_dict['KEY_BLOCK_ST'] = keyblock_strategy
    def get_keyblock_strategy(self):
        return self._var_dict['KEY_BLOCK_ST']
    # search strategy
    def set_openblock_strategy(self, opblk_st):
        self._var_dict['OPEN_BLOCK_ST'] = opblk_st
    def get_openblock_strategy(self):
        return self._var_dict['OPEN_BLOCK_ST']
    def set_openprblm_strategy(self, opprblm_st):
        self._var_dict['OPEN_PROBLEM_ST'] = opprblm_st
    def get_openprblm_strategy(self):
        return self._var_dict['OPEN_PROBLEM_ST']
    # gas
    def set_total_gas(self, total_gas):
        self._var_dict['total_gas'] = total_gas
    def get_total_gas(self):
        return self._var_dict['total_gas']
    def set_once_gas(self, once_gas):
        self._var_dict['once_gas'] = once_gas
    def get_once_gas(self):
        return self._var_dict['once_gas']

    # 全局自增prblm_id管理
    def key_id_generator(self):
        self._var_dict['KEY_PRBLM_NUMBER'] = self._var_dict['KEY_PRBLM_NUMBER'] + 1
        self._prblm_id_center[self._var_dict['KEY_PRBLM_NUMBER']] = 0
        return self._var_dict['KEY_PRBLM_NUMBER']
    def autoinc_prblm_id_generator(self, key_id):
        self._prblm_id_center[key_id] = self._prblm_id_center[key_id] + 1
        return self._prblm_id_center[key_id] + 1
    def reset_id_center(self):
        self._prblm_id_center.clear()

    # gas管理
    def init_gases(self, prblm_pool:list[data.lpprblm.LpPrblm]):
        if prblm_pool[0].fix_pid is None:
            [setattr(prblm, 'fix_id', i) for i, prblm in enumerate(prblm_pool)]
        for p in prblm_pool:
            self._gases[p.fix_pid] = self.get_total_gas()
    
    def consume_gas_once(self, fix_pid:int):
        """
        :return: rest_gas(int) 该问题剩余的gas
        :return: consume_success(bool) 是否成功获得gas奖励
        """
        current_gas = self._gases[fix_pid]
        if current_gas <= 0:
            return current_gas, False
        
        rest_gas = max(0, current_gas - self.get_once_gas())
        self._gases[fix_pid] = rest_gas
        return rest_gas, True
    
    def consume_gas_multi(self, fix_pid:int, prblm_count:int):
        """
        :return: rest_gas(int) 该问题剩余的gas
        :return: consume_success(bool) 是否成功获得gas奖励
        """
        current_gas = self._gases[fix_pid]
        if current_gas <= 0:
            return current_gas, False
        rest_gas = max(0, current_gas - self.get_once_gas() * prblm_count)
        self._gases[fix_pid] = rest_gas
        return rest_gas, True
    
    def get_rest_gas(self, fix_pid):
        return self._gases[fix_pid]
    
    def is_gas_used_up(self, fix_pid:int):
        return self._gases[fix_pid] <= 0

BACKGROUND = Background()
