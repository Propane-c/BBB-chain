import json
import sys
from collections import defaultdict
from dataclasses import asdict, astuple, dataclass

from background import Background

# from miner import Miner
from data import Block, Chain
from data import LpPrblm


@dataclass
class EvaResult:
    var_num:int
    miner_num:int
    difficulty:int
    safe_thre:float
    solve_prob:float
    openblk_st:str
    openprblm_st:str
    # solving process
    lowerbounds:dict # {round:lb}
    ubdata:list
    # solve round
    solve_rounds:dict # {keyblock.name: 求解轮数}
    # subpair nnum
    subpair_nums:dict # {keyblock.name: 子问题对数量}
    acp_subpair_nums:dict # {keyblock.name: 被接受的子问题对数量}
    subpair_unpub:dict# {keyblock.name: 子问题对数量}
    # miniblock forkrate
    mb_nums:dict  # {keyblock.name: miniblock数量}
    accept_mb_nums:dict # {keyblock.name: 被接受的miniblock数量}
    mb_forkrates:dict # {keyblock.name: miniblock分叉率}
    total_mb_forkrate:float
    # keyblock forkrate 
    kb_forkrate:float
    kb_forknum:int
    kb_num:int
    # growth rate
    mb_growths:dict
    mb_grow_proc:list
    # block time
    mb_times:list
    unpub_times:list
    fork_times:list
    kb_times:list
    # attack
    attack_num:int
    success_num:int
    success_rate:float
    atklog_depth:dict
    atklog_bname:dict
    # 攻击者区块的数量和比例
    advblock_nums:dict
    accept_advblock_nums:dict
    adv_rates:dict # {keyblock.name: 攻击者区块占所有区块的比例}
    accept_adv_rates:dict # {keyblock.name: 攻击者区块占接受链的比例}

@dataclass
class UbData():
    miner:int
    block:str # the outer block
    round:int
    bround:int
    pname:str
    pre_pname:str
    ub:float
    fathomed:bool
    allInteger:bool
    isFork:bool

class Evaluation(object):
    def __init__(self, background:Background, recordSols:bool = False):
        self.background = background
        self.result:EvaResult = None
        self.recordSols = recordSols
        self.feasi_kbs = []
        # fig 1 solving process
        self.cur_rlxsol = 0 
        self.cur_ub = sys.maxsize # 全局的最大下界
        self.relax_sols = defaultdict(list) # {round:[rlx_sols for cur round]}
        self.upperbounds = defaultdict(int) # {round:lb}
        self.ubdata:list[tuple] = []
        # fig 2 keyblock inter-block time
        self.kb_block_times = []
        # fig 2 number of solving round
        self.solve_rounds = {} # {keyblock.name: 求解轮数}
        # fig 3 number of sub-problem pairs
        self.subpair_nums = {} # {keyblock.name: 子问题对数量}
        self.acp_subpair_nums = {}
        self.subpair_unpub = defaultdict(int) # {keyblock.name: 未发布的子问题对数量}
        # fig 4 fork rate of miniblock
        self.mb_nums = {}
        self.accept_mb_nums = {}
        self.miniblock_forkrates = {}
        self.total_mb_forkrate = 0
        # fig 5 fork rate of keyblock
        self.keyblocks = {} # {keyblock.name: [next_keyblocks]}
        self.keyblock_forkrate = 0
        self.kb_num = 0
        self.kb_forknum = 0
        # fig 8 steal_attack
        self.attack_num = 0
        self.success_num = 0
        self.success_rate = 0
        self.atklog_depth = defaultdict(
            lambda : {"attack_num":0,"success_num":0,"success_rate":0})
        self.atklog_bname = defaultdict(
            lambda : {"depth":0, "theory":0, "attack_num":0,
                      "success_num":0, "success_rate":0})
        # 攻击者区块的数量和比例
        self.advblock_nums = defaultdict(int)
        self.accept_advblock_nums = defaultdict(int)
        self.accept_adv_rates = defaultdict(float)
        self.adv_rates = defaultdict(float)
        # fig 6 miniblock inter-block time
        self.mb_times = []
        self.mb_times2 = []
        self.unpub_times = []
        self.fork_times = []
        # growth rate
        self.mb_growths = defaultdict(float) # {keyblock.name: growth_rate}
        self.kb_growths = 0
        self.mb_grow_proc = [] # {keyblock.name: grow_proc}
    
    

    def get_feasi_kbs(self, chain:Chain):
        if len(self.feasi_kbs) > 0:
            return self.feasi_kbs
        self.feasi_kbs = chain.get_feasible_keyblocks()
        return self.feasi_kbs

    def get_mb_block_times(self, chain:Chain):
        """
        获取miniblock times: 和连接的上一个mini-block的时间差
        """
        feasi_kbs = self.get_feasi_kbs(chain)
        if len(feasi_kbs) == 0:
            return []
        for kb in feasi_kbs:
            if kb.keyfield.key_tx is None:
                continue
            mbs = chain.get_acpmbs_after_kb_and_label_forks(kb)
            for mb in mbs:
                if ((not mb.pre.iskeyblock) and 
                    mb.pre.get_keyid() != mb.get_keyid()):
                    continue
                mb_time = mb.get_block_time_w_pre()
                self.mb_times.append(mb_time)
        return self.mb_times

    def get_mb_block_times2(self, chain:Chain):
        """
        获取miniblock times: 和上一个发布出来的mini-block的时间差
        """
        feasi_kbs = self.get_feasi_kbs(chain)
        if len(feasi_kbs) == 0:
            return []
        for kb in feasi_kbs:
            if kb.keyfield.key_tx is None:
                continue
            mbs = chain.get_acpmbs_after_kb_and_label_forks(kb)
            sorted_mbs:list[Block] = sorted(mbs, key = lambda x : x.get_timestamp())
            pre_timestamp = kb.get_timestamp()
            mb_count = 0
            for mb in sorted_mbs:
                mb_count+=1
                if ((not mb.pre.iskeyblock) and 
                    mb.pre.get_keyid() != mb.get_keyid()):
                    continue
                if mb.isFork:
                    continue
                cur_mbtime = mb.get_timestamp()
                self.mb_times2.append(cur_mbtime-pre_timestamp)
                pre_timestamp = cur_mbtime
                self.mb_grow_proc.append(cur_mbtime/mb_count)
                


        # if len(timestamps) == 0:
        #     warnings.warn("get timestamps failed!")
        # timestamps = sorted(timestamps)
        # timestamps.insert(0, 0)
        # for i in range(1, len(timestamps)-1):
        #     j = i
        #     while timestamps[i+1] == timestamps[j] and j > 0:
        #         j -= 1
        #     self.mb_times2.append(timestamps[i+1]-timestamps[j])

    # def get_mb_growth_rate(self,chain:Chain):
    #     feasi_kbs = self.get_feasi_kbs(chain)
    #     for kb in feasi_kbs:
    #         if kb.keyfield.key_tx is None:
    #             continue
            
                
    def record_relaxed_solutions(self, miner, round:int):
        """
        记录每一轮所有矿工的松弛解
        """
        conss_un = miner.consensus.upper_bound
        self.cur_rlxsol = conss_un if conss_un != sys.maxsize else 0
        self.relax_sols[round].append(self.cur_rlxsol)

    def get_ubs(self, chain:Chain):
        feasi_kbs = self.get_feasi_kbs(chain)
        if len(feasi_kbs) == 0:
            return
        for kb in feasi_kbs:
            if len(kb.keyfield.next_kbs)==0 :
                continue
            # if not kb.get_fthmstat():
            #     continue
            kp = kb.get_keyprblm()
            kub = UbData(kb.get_miner_id(), kb.name, kp.timestamp, 
                         kb.get_timestamp(), kp.pname, "None", kp.z_lp,
                         False, kp.all_integer(),  False)
            self.ubdata.append(astuple(kub))
            acp_mbs = chain.get_acpmbs_after_kb_and_label_forks(kb)
            mbs = chain.get_mbs_after_kb(kb)
            for mb in mbs:
                p:LpPrblm
                for p in mb:  
                    allInt = p.all_integer() if p.lb_prblm is None else False
                    # fathomed = p.fathomed if not allInt else  False
                    pub = UbData(mb.get_miner_id(), mb.name, p.timestamp, 
                                 mb.get_timestamp(),p.pname, p.pre_pname, p.z_lp, 
                                 p.fathomed, allInt, mb.isFork)
                    self.ubdata.append(astuple(pub))
                    
    def record_upperbound(self, miner, round:int):
        """
        记录全局的最大上界
        """
        conss_ub = miner.consensus.upper_bound
        self.cur_ub = conss_ub if conss_ub < self.cur_ub else self.cur_ub 
        self.upperbounds[round] = self.cur_ub
        # self.lb_perminer[miner.Miner_ID]

    def get_solving_rounds_kbtimes_mbgrowth(self, chain:Chain):
        """
        获取每个keyblock被解决的总轮数
        """
        feasi_kbs = self.get_feasi_kbs(chain)
        if len(feasi_kbs) == 0:
            return {}
        for kb in feasi_kbs:
            if len(kb.keyfield.next_kbs)==0 :
                continue
            if not kb.get_fthmstat():
                continue
            solve_round = kb.get_kb_time_w_next()
            self.solve_rounds.update({kb.name:solve_round})
            self.kb_block_times.append(solve_round)
            self.mb_growths[kb.name] = solve_round / len(chain.get_acpmbs_after_kb_and_label_forks(kb))
        return self.solve_rounds

        
    def get_subpair_nums(self, chain:Chain):
        """
        获取链上所有keyblock对应的子问题对数量
        """
        feasi_kbs = self.get_feasi_kbs(chain)
        if len(feasi_kbs) == 0:
            return {}
        for kb in feasi_kbs:
            if kb.keyfield.key_tx is None:
                continue
            mbs = chain.get_mbs_after_kb(kb)
            acp_mbs = chain.get_acpmbs_after_kb_and_label_forks(kb)
            if len(mbs) == 0 or acp_mbs == 0:
                continue
            subpair_num = 0
            acp_subpair_num = 0
            for mb in mbs:
                subpair_num += len(mb.minifield.subprblm_pairs)
            for acpmb in acp_mbs:
                acp_subpair_num += len(acpmb.minifield.subprblm_pairs)
            self.subpair_nums.update({kb.name: subpair_num})
            self.acp_subpair_nums.update({kb.name: acp_subpair_num})
        return self.subpair_nums, self.acp_subpair_nums
    
    
    def merge_subpair_num(self):
        """
        将未发布的子问题对和已发布的子问题对合并
        """
        subpair_unpub = defaultdict(int)
        for kb_name, num in self.subpair_nums.items():
            subpair_unpub[kb_name] = num + self.subpair_unpub[kb_name]
        return dict(subpair_unpub)

    def record_unpub_pair(self, key_name, subpair_unpub:int, 
                        unpub_pair_time:int = None, 
                        unpub_subs:list[LpPrblm] = None, miner_id:int = None):
        """
        记录已算出但未发布出来的sub-problem数量
        """
        self.subpair_unpub[key_name] += subpair_unpub
        if unpub_pair_time is not None:
            self.unpub_times.append(unpub_pair_time)
        if not self.recordSols:
            return
        for p in unpub_subs:
            p:LpPrblm
            allInt = p.all_integer() if p.lb_prblm is None else False
            self.ubdata.append(astuple(UbData(miner_id, "None", p.timestamp, -1, 
                                              p.pname, p.pre_pname,
                                              p.z_lp, p.fathomed, allInt,  False)))

    def record_fork_times(self, fork_time:int = None):
        """记录产生fork的时间点"""
        self.fork_times.append(fork_time)

    def cal_growth_rate_process(self, chain:Chain):
        ...


    def cal_miniblock_fork_rate(self, chain:Chain):
        """
        保存keyblock主链上各个keyblock下的miniblock和 accepted miniblock数量
        并计算分叉率: 
        (sum(mb_nums) - sum(accepted_mb_nums)) / sum(mb_nums)
        """
        total_mb_num = 0
        total_accepted_mb_num = 0
        feasi_kbs = self.get_feasi_kbs(chain)
        if len(feasi_kbs) == 0:
            return {}
        for kb in feasi_kbs:
            if len(kb.next) == 0:
                continue
            if kb.keyfield.key_tx is None:
                continue
            if kb.keyfield.key_tx.data.fthmd_state is False:
                continue
            mbs = chain.get_mbs_after_kb(kb)
            acp_mbs = chain.get_acpmbs_after_kb_and_label_forks(kb)
            total_mb_num += len(mbs)
            total_accepted_mb_num += len(acp_mbs)
            self.mb_nums.update({kb.name:len(mbs)})
            self.accept_mb_nums.update({kb.name:len(acp_mbs)})
            if len(mbs)>0:
                mb_forkrate = (len(mbs)- len(acp_mbs))/len(mbs)
            self.miniblock_forkrates.update({kb.name:mb_forkrate})
        if total_mb_num > 0:
            self.total_mb_forkrate = (total_mb_num-total_accepted_mb_num)/total_mb_num
        return self.miniblock_forkrates, self.total_mb_forkrate
    
    def cal_keyblock_fork_rate(self, chain:Chain):
        """
        保存链上的keyblock以及它们的下一个keyblock
        并计算分叉率: 
        kb_fork_num / total_kb_num
        """
        kbs = chain.get_keyblocks_pref()
        self.kb_num = len(kbs)
        self.kb_forknum = 0
        for kb in kbs:
            self.keyblocks.update({kb.name: [b.name for b in kb.keyfield.next_kbs]
                                if len(kb.keyfield.next_kbs)>0 else []})
            if len(kb.keyfield.next_kbs) > 1:
                self.kb_forknum += (len(kb.keyfield.next_kbs) - 1)
        self.keyblock_forkrate = self.kb_forknum / self.kb_num
        return self.keyblock_forkrate

    # steal attack
    def record_atk(self, attack:bool, mb_depth:int, block_name:str, theory:float):
        """
        记录每一次攻击行为
        """
        if attack:
            self.attack_num += 1
            self.atklog_depth[mb_depth]["attack_num"] += 1
            self.atklog_bname[block_name]["attack_num"] += 1
            if self.atklog_bname[block_name]["depth"] == 0:
                self.atklog_bname[block_name]["theory"] = theory
            if self.atklog_bname[block_name]["depth"] == 0:
                self.atklog_bname[block_name]["depth"] = mb_depth
    
    def record_success_atk(self, attack_success:bool, mb_depth:int, block_name):
        """
        记录每一次成功的攻击
        """
        if attack_success:
            self.success_num += 1
            self.atklog_depth[mb_depth]["success_num"] += 1
            self.atklog_bname[block_name]["success_num"] += 1
        
    def cal_steal_success_rate(self):
        """
        计算攻击成功的概率
        """
        if self.attack_num > 0:
            self.success_rate = self.success_num/self.attack_num
            for mb_depth in self.atklog_depth.keys():
                success_num = self.atklog_depth[mb_depth]["success_num"]
                attack_num = self.atklog_depth[mb_depth]["attack_num"]
                self.atklog_depth[mb_depth]["success_rate"] = success_num / attack_num

    
    def cal_adversary_block_rate(self, chain:Chain):
        """
        记录攻击者产生的区块在接受链和所有miniblock中的数量和比例
        """
        feasi_kbs = self.get_feasi_kbs(chain)
        for kb in feasi_kbs:
            if len(kb.next) == 0:
                continue
            if kb.keyfield.key_tx is None:
                continue
            if kb.keyfield.key_tx.data.fthmd_state is False:
                continue
            advblock_num = 0
            accept_advblock_num = 0
            mbs = chain.get_mbs_after_kb(kb)
            accepted_mbs = chain.get_acpmbs_after_kb_and_label_forks(kb)
            for mb in mbs:
                if mb.isAdversary:
                    advblock_num += 1
            for amb in accepted_mbs:
                if amb.isAdversary:
                    accept_advblock_num += 1
            self.advblock_nums[kb.name] = advblock_num
            self.accept_advblock_nums[kb.name] = accept_advblock_num
            self.adv_rates[kb.name] = advblock_num/len(mbs)
            self.accept_adv_rates[kb.name] = accept_advblock_num/len(accepted_mbs)
        
    
    def cal_packaged_results(self, chain):
        """
        打包计算所有评估指标
        """
        self.get_solving_rounds_kbtimes_mbgrowth(chain)
        self.get_subpair_nums(chain)
        self.cal_keyblock_fork_rate(chain)
        self.cal_miniblock_fork_rate(chain)
        self.get_mb_block_times2(chain)
        self.cal_steal_success_rate()
        self.cal_adversary_block_rate(chain)


    def collect_evaluation_results(self):
        """收集所有评估结果, 并以EvaResult的对象返回"""
        miner_num = self.background.get_miner_num()
        diffculty = self.background.get_bb_difficulty()
        var_num = self.background.get_var_num()
        safe_thre= self.background.get_safe_thre()
        solve_prob = self.background.get_solve_prob()
        openblk_st = self.background.get_openblock_strategy()
        openprblm_st = self.background.get_openprblm_strategy()
        self.result = EvaResult(
            var_num = var_num,
            miner_num = miner_num,
            difficulty = diffculty,
            safe_thre = safe_thre,
            solve_prob = solve_prob,
            openblk_st = openblk_st,
            openprblm_st = openprblm_st,
            lowerbounds = dict(self.upperbounds),
            ubdata = self.ubdata,
            solve_rounds = self.solve_rounds,
            subpair_nums = self.subpair_nums,
            acp_subpair_nums = self.acp_subpair_nums, 
            # subpair_unpub = dict(self.subpair_unpub),
            subpair_unpub = self.merge_subpair_num(),
            mb_nums = self.mb_nums,
            accept_mb_nums = self.accept_mb_nums,
            mb_forkrates = self.miniblock_forkrates,
            total_mb_forkrate = self.total_mb_forkrate,
            kb_forkrate = self.keyblock_forkrate,
            kb_num = self.kb_num,
            kb_forknum = self.kb_forknum,
            mb_growths=dict(self.mb_growths),
            mb_grow_proc=self.mb_grow_proc,
            mb_times = self.mb_times2,
            unpub_times = self.unpub_times,
            fork_times=self.fork_times,
            kb_times = self.kb_block_times,
            attack_num = self.attack_num,
            success_num = self.success_num,
            success_rate = self.success_rate,
            atklog_depth= dict(self.atklog_depth),
            atklog_bname = dict(self.atklog_bname),
            advblock_nums = dict(self.advblock_nums),
            accept_advblock_nums = dict(self.accept_advblock_nums),
            accept_adv_rates = dict(self.accept_adv_rates),
            adv_rates = dict(self.adv_rates),
        )
        return self.result

    def get_result_dict(self):
        return asdict(self.result)

    def save_results_to_json(self, pool_path):
        result_path = self.background.get_result_path()
        miner_num = self.background.get_miner_num()
        diffculty = self.background.get_bb_difficulty()
        var_num = self.background.get_var_num()
        json_name = f'm{miner_num}d{diffculty}v{var_num}evaluation results.json'
        if pool_path is not None:
            json_name = f'p{pool_path.stem}m{miner_num}d{diffculty}evaluation results.json'
        with open(result_path / json_name, 'w+') as f:
                result_dict = self.get_result_dict()
                json_res = json.dumps(result_dict)
                f.write(json_res)




if __name__ == "__main__":
    d = defaultdict(list)
    d[2].append(1)
    d[3].append(2)
    print(d)
    # for k in d.keys():
    #     print(k,type(k))