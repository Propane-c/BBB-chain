import copy
import json
import logging
import math
import random
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict

from matplotlib import pyplot as plt

import lpprblm
import network
from background import Background
from chain import Block, Chain, NewBlocks
from consensus import BranchBound
from evaluation import Evaluation
from miner import Miner

logger = logging.getLogger(__name__)
class VirtualAttacker():
    '''代表攻击者集团的虚拟矿工对象， 以Adverchain作为本地链，与全体攻击者共享共识参数'''
    ADVERMINER_ID = -2 # Miner_ID默认为ADVERMINER_ID
    def __init__(self, context:Background):
        '''重写初始化函数，仅按需初始化Miner_ID、isAdversary以及共识对象'''
        self.miner_id = VirtualAttacker.ADVERMINER_ID #矿工ID
        self.is_adversary = True
        self.vchain = Chain(context)
        self.consensus = BranchBound(context, self.miner_id)
        self.merged_rcvtape:list[NewBlocks] = []
        
class Attack(metaclass=ABCMeta): 
    @abstractmethod
    def renew(self):
        # 更新adversary中的所有区块链状态：基准链 矿工状态(包括输入和其自身链 )
        pass

    @abstractmethod
    def clear(self):
        # clear the input tape and communcation tape
        pass

    @abstractmethod
    def adopt(self):
        # Adversary adopts the newest chain based on tthe adver's chains
        pass

    @abstractmethod
    def wait(self):
        # Adversary waits, and do nothing in current round.
        pass

    @abstractmethod
    def giveup(self):
        # Adversary gives up current attacking, like mining the blocks based on formal chain.
        pass

    @abstractmethod
    def match(self):
        # Although adversary did not attack successfuly, it broadcast the block at the same height of the main chain.
        pass

    @abstractmethod
    def mine(self):
        pass


class default_attack_mode(metaclass = ABCMeta):
    def __init__(self, context: Background,
                evaluation: Evaluation,
                adversary_miner: list[Miner],
                global_chain: Chain, 
                outer_network: network):
        self.context = context
        self.evaluation = evaluation
        self.attackers: list[Miner] = adversary_miner
        # 初始当前矿工代表
        self.current_miner = self.attackers[0]
        self.global_chain: Chain = global_chain
        # 攻击链 攻击者采用攻击手段挖出的块暂时先加到这条链上
        self.adver_chain = copy.deepcopy(self.global_chain) 
        # 基准链 攻击者参考的链, 始终是以adversary视角出发的最新的链
        self.base_chain = copy.deepcopy(self.global_chain)
        self.network: network.Network = outer_network
        # 攻击者总算力
        self.total_q = sum([attacker.q for attacker in self.attackers]) 
        for ad_miner in self.attackers:
            # 重新设置adversary的 q 和 blockchain，原因在 mine_randmon_miner 部分详细解释了
            ad_miner.q = self.total_q
            ad_miner.local_chain.add_block_copy(self.base_chain.lastblock)
        # 虚拟攻击者，整合攻击者集团的算力进行挖矿
        self.vattacker = VirtualAttacker(self.context)
        self.vattacker.vchain = self.adver_chain
        # 攻击历史
        self.attack_history = []
        self.atklog_mbname = defaultdict(int)
        self.atklog_success = defaultdict(int)
        self.atk_theory = defaultdict(lambda: 1)
        # 状态转移
        self.state_trans = None


    def renew(self, round): 
        """更新adversary中的所有区块链状态：基准链和矿工状态(包括输入和其自身链 )"""
        for ad_miner in self.attackers:
            chain_update, update_index = ad_miner.fork_choice(round) 
            self.base_chain.add_block_copy(chain_update.lastblock) # 如果存在更新将更新的区块添加到基准链上 
            #self.local_record.add_block_copy(chain_update.lastblock) # 同时 也将该区块同步到全局链上
        newest_block = self.base_chain.lastblock
        return newest_block
    
    def clear(self): 
        """清除矿工的input tape和communication tape"""
        for temp_miner in self.attackers:
            temp_miner.input_tape = []  # clear the input tape
            temp_miner.receive_tape = []  # clear the communication tape
        
    def mine_steal(self, round:int):
        """
        攻击入口
        """
        self.merge_receive_tape()
        stealed_newmbs = self.steal_attack(round)
        # print('attack2', stealed_newmbs)
        return stealed_newmbs
    
    def merge_receive_tape(self):
        """
        整合所有攻击者的receive tape到virtual_attacker
        """
        merged_rcvtape:list[NewBlocks] = []
        for attacker in self.attackers:
            if len(attacker.receive_tape) == 0:
                continue
            for newblock in attacker.receive_tape:
                if len(merged_rcvtape) == 0:
                    merged_rcvtape.append(newblock)
                if (newblock.miniblock.name not in 
                    [newb.miniblock.name for newb in merged_rcvtape]):
                    merged_rcvtape.append(newblock)
        self.vattacker.merged_rcvtape = merged_rcvtape

    def steal_attack(self, round:int):
        """
        启动攻击
        """
        attack_success = False
        stealed_newmbs:list[NewBlocks] = []
        if len(self.vattacker.merged_rcvtape) == 0:
            return stealed_newmbs
        for newblock in self.vattacker.merged_rcvtape:
            # 目前只偷miniblock的答案
            if newblock.iskeyblock:
                continue
            honest_mb = newblock.miniblock
            # 如果已经攻击过了，就不再攻击
            if (honest_mb.name in self.attack_history or 
                honest_mb.minifield.pre_pname in self.attack_history):
                continue
            self.attack_history.append(honest_mb.name)
            self.attack_history.append(honest_mb.minifield.pre_pname)
            # 计算攻击成功理论值
            attack_theory = self.cal_theory_rate_simple(honest_mb)
            # 开始攻击
            attack_iter = 1
            for _ in range(attack_iter):
                # 有多少attacker实施多少次攻击
                mb_depth = honest_mb.get_soltree_depth()
                self.evaluation.record_atk(True, mb_depth, honest_mb.name, attack_theory)
                self.atklog_mbname[honest_mb.name] += 1
                attack_success = False
                for attacker in self.attackers:
                    steal_success = self.steal_miniblock(round, attacker, honest_mb)
                    attack_success = attack_success or steal_success
                    if not steal_success:
                        continue
                    else:
                        # 针对一个块实施多次攻击时，是为了测试攻击成功概率，不产生区块
                        if attack_iter > 1:
                            break
                        # 如果攻击成功则产生新的区块
                        steal_newmb = self.mb_reassemble(round, attacker, honest_mb)
                        stealed_newmbs.append(steal_newmb)
                        self.attack_history.append(steal_newmb.miniblock.name)
                        break
                if attack_success:
                    self.evaluation.record_success_atk(True, mb_depth, honest_mb.name)
                    self.atklog_success[honest_mb.name] += 1
        if len(stealed_newmbs) == 0:
            return stealed_newmbs
        
        for nb in stealed_newmbs:
            for attacker in self.attackers:
                attacker.receive_tape.append(nb)
            miner_id = nb.miniblock.blockhead.miner_id
            self.network.access_network(nb, miner_id, round)
        self.vattacker.merged_rcvtape = []
        # print('attack', stealed_newmbs)
        return stealed_newmbs
            

    def steal_miniblock(self, round:int, attacker:Miner, honest_mb:Block):
        """
        尝试偷取miniblock的答案
        """
        logger.info(f"round{round}: attacker{attacker.miner_id} "
                    f"stealing answers from {honest_mb.name}")
        for (p1, _) in honest_mb.minifield:
            if not self.steal_answer(p1, honest_mb):
                logger.info(f"round{round}: attacker{attacker.miner_id} "
                            f"stealing answers from {honest_mb.name} Failed!")
                return False
        logger.info(f"round{round}: attacker{attacker.miner_id} "
                    f"stealing answers from {honest_mb.name} Success!")
        return True
    
    def get_prev_hprblm(self, cur_prblm:lpprblm.LpPrblm, honest_mb:Block):
        """
        获取该问题连接的前一个问题
        """
        (root1, _) = honest_mb.minifield.root_pair
        if cur_prblm.pname == root1.pname:
            return honest_mb.minifield.pre_p
        for prblm in honest_mb:
            if prblm.pname == cur_prblm.pre_pname:
                return prblm
        return None

    def cal_theory_rate(self, honest_mb:Block):
        """
        `warning`:弃用, 使用简化版 cal_theory_rate_simple

        计算攻击成功理论值
        """
        warnings.warn("cal_theory_rate is deprecated", DeprecationWarning)
        for (p1, _) in honest_mb.minifield:
            prev_prblm = self.get_prev_hprblm(p1, honest_mb)
            if prev_prblm is None:
                continue
            # 前一个问题没有解
            if prev_prblm.x_lp is None:
                continue
            not_integer_idx = [idx for idx, x in enumerate(prev_prblm.x_lp)
                            if not x.is_integer()]
            # 前一个问题全是整数解
            if len(not_integer_idx)==0:
                continue
            # 计算理论值
            self.atk_theory[honest_mb.name] *= (1/len(not_integer_idx))
    
    def cal_theory_rate_simple(self, honest_mb:Block):
        """
        计算攻击成功理论值，简化版
        """
        theory_rate = 1
        for (p1, _) in honest_mb.minifield:
            theory_rate *= (1 / p1.pre_rest_x)
        self.atk_theory[honest_mb.name] = theory_rate
        return theory_rate

    def save_draw_final_success_rates(self):
        """
        计算所有区块的仿真攻击成功概率
        """
        attack_rates = defaultdict(lambda: {
            "attack_count":0, "success_count":0, "success_rate":0, "theory":0})
        for bname in self.atk_theory.keys():
            attack_rates[bname]["attack_count"] = self.atklog_mbname[bname]
            attack_rates[bname]["success_count"] = self.atklog_success[bname]
            attack_rates[bname]["theory"] = self.atk_theory[bname]
            attack_rates[bname]["success_rate"] = \
                (self.atklog_success[bname]/self.atklog_mbname[bname]
                 if bname in self.atklog_mbname else 0)
        for key, values in attack_rates.items():
            values['success_rate'] = math.log(values['success_rate']) if values['success_rate'] > 0 else 0
            values['theory'] = math.log(values['theory'])
        # 将attack_rates保存到JSON文件
        result_path = self.context.get_result_path()
        with open(result_path / 'atk_rates.json', 'w') as json_file:
            json.dump(attack_rates, json_file)

        sorted_attack_rates = dict(sorted(attack_rates.items(), key=lambda item: item[1]['theory'], reverse=True))
        attacks = range(len(sorted_attack_rates.keys()))
        success_rates = [values['success_rate'] for values in sorted_attack_rates.values()]
        theory_values = [values['theory'] for values in sorted_attack_rates.values()]
        plt.figure(figsize=(10, 6))
        plt.bar(attacks, success_rates, label='Attack Success Rate', color='orange', alpha=0.7)
        plt.plot(attacks, theory_values, label='Theory', color='#1f77b4', alpha=0.7, linestyle = "--")
        plt.xlabel('Blocks')
        plt.ylabel('Success rate of plagiarism(log)')
        plt.legend()
        plt.show()
        return attack_rates

    def steal_answer(self, cur_prblm:lpprblm.LpPrblm, honest_mb:Block):
        """
        偷答案，看能否产生和原问题一样的idx
        """
        # 获取前一个问题
        prev_prblm = self.get_prev_hprblm(cur_prblm, honest_mb)
        if prev_prblm is None:
            logger.warning(f"not found the previous problem {cur_prblm.pre_pname} "
                           f"of problem{cur_prblm.pname} in {honest_mb.name}!")
            return False
        # 前一个问题没有解，攻击成功
        if prev_prblm.x_lp is None:
            return True
        not_integer_idx = [idx for idx, x in enumerate(prev_prblm.x_lp)
                           if not x.is_integer()]
        # 前一个问题全是整数解，攻击成功
        if len(not_integer_idx)==0:
            return True
        # 产生了前一个问题相同的x_nk，攻击成功
        idx = random.choice(not_integer_idx)
        logger.info(f"stealing x_nk is {idx}, honest is {cur_prblm.x_nk} "
                    f"with optional x {not_integer_idx}")
        if idx == cur_prblm.x_nk:
            return True
        return False

    def mb_reassemble(self,  round:int, attacker:Miner, honest_mb:Block):
        """
        拷贝和重组miniblock
        """
        copy_mb = copy.deepcopy(honest_mb)
        copy_mb.blockhead.miner_id = attacker.miner_id
        copy_mb.blockhead.timestamp = round
        copy_mb.name = f'B{str(self.context.get_block_number())}'
        copy_mb.blockhead.blockhash = copy_mb.name
        # 重命名问题id
        coor = defaultdict(int)
        coor[copy_mb.minifield.pre_pname[0][1]] = copy_mb.minifield.pre_pname[0][1]
        for (p1,p2) in copy_mb.minifield:
            key_id = p1.pname[0][0]
            sub_id = self.context.autoinc_prblm_id_generator(key_id)
            coor[p1.pname[0][1]] = sub_id
            p1.pname = ((key_id, sub_id), p1.pname[1])
            p2.pname = ((key_id, sub_id), p2.pname[1])
        # 重新建立问题间连接
        for (p1,p2) in copy_mb.minifield:
            pre_key_id = p1.pre_pname[0][0]
            pre_sub_id = coor[p1.pre_pname[0][1]]
            p1.pre_pname = ((pre_key_id, pre_sub_id), p1.pre_pname[1])
            p2.pre_pname = ((pre_key_id, pre_sub_id), p1.pre_pname[1])
        # 重新建立连接
        copy_mb.pre = honest_mb.pre
        copy_mb.isAdversary = True
        for prblm in honest_mb.pre:
            if prblm.pname == honest_mb.minifield.pre_pname:
                copy_mb.minifield.pre_p = prblm
        copy_newmb = NewBlocks(False, copy_mb, None, [], None)
        return copy_newmb
    
    def add_miniblock_to_local_chain(attacker:Miner, rcv_mb:Block):
        """
        将miniblcok复制后添加到本地链上
        return: 复制过后的区块
        """
        copyblk = copy.deepcopy(rcv_mb)
        add_success = False
        block_to_link = \
            attacker.local_chain.search_forward_by_hash(rcv_mb.blockhead.prehash)
        if block_to_link is None:
            logger.warning(f"Attacker {attacker.miner_id}: "
                           f"add block {rcv_mb.name} failed, "
                           f"not found {rcv_mb.blockhead.prehash}")
            return copyblk, add_success
        add_success = True
        attacker.local_chain.add_block_direct(copyblk, block_to_link)
        # 与该区块branch的子问题建立连接
        for prblm in block_to_link:
            if prblm.pname == rcv_mb.minifield.pre_pname:
                copyblk.minifield.pre_p = prblm
        return copyblk, add_success    


if __name__ == "__main__":
    POW = "pow"
    W_MINI = "withmini"
    print('powwithmini' == POW+W_MINI)