import sys

sys.path.append("d:\\Files\\gitspace\\chain-xim")
import copy  # noqa
import logging  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

import data.lpprblm as lpprblm  # noqa: E402
from background import Background  # noqa: E402
from data import Block, BlockHead, Chain, NewBlocks  # noqa: E402
from functions import hashsha256  # noqa: E402, F401
from data.lpprblm import IncConstr, LpPrblm  # noqa: E402
from data.txpool import TxPool  # noqa: E402
from evaluation import Evaluation

logger = logging.getLogger(__name__)

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

# var 选择策略
VAR_SPEC = "var_specific"
VAR_RAND = "var_random"
 
class SolvingPair(object):
    def __init__(self, p1: LpPrblm = None, p2: LpPrblm = None, key_lp:LpPrblm = None,
                 solve_prob_init=0.01):
        self.p1 = p1
        self.p2 = p2
        self.key_lp = key_lp
        self.G_ub1, self.h_ub1 = p1.get_Gub_hub(key_lp)
        self.G_ub2, self.h_ub2 = p2.get_Gub_hub(key_lp)
        self.success1 = False
        self.success2 = False
        self.solve_prob_init = solve_prob_init
        self.solve_prob = self.solve_prob_init

    def get_unsolved_prblms(self):
        if self.success1 and self.success2:
            return None, None
        elif (not self.success1) and self.success2:
            return self.p1, None
        elif self.success1 and (not self.success2):
            return None, self.p2
        else:
            return self.p1, self.p2

    def try_solve_cur_prblms(self, round):
        '''Try to solve the subprblms'''
        solve_finished = False
        p1, p2 = self.get_unsolved_prblms()
        if p1 is not None:
            self.success1 = lpprblm.solve_lp(
                p1, self.key_lp.c, self.G_ub1, self.h_ub1, 
                self.key_lp.A_eq, self.key_lp.b_eq, 
                self.key_lp.bounds, self.solve_prob)
            if self.success1:
                p1.timestamp = round
        elif p2 is not None:
            self.success2 = lpprblm.solve_lp(
                p2, self.key_lp.c, self.G_ub2, self.h_ub2, 
                self.key_lp.A_eq, self.key_lp.b_eq, 
                self.key_lp.bounds, self.solve_prob)
            if self.success2:
                p2.timestamp = round
        # whether all problems are solved
        solve_finished = self.success1 and self.success2
        # update the solve_prob
        if solve_finished is False:
            self.solve_prob += 0
        return solve_finished


class BranchBound(object):
    def __init__(self, background: Background, chain: Chain, miner_id=None, evaluation:Evaluation = None):
        self.local_chain = chain
        self.background = background
        self.miner_id = miner_id
        self.evaluation = evaluation
        self.round = 0
        self.LOG_PREFIX = f"round {self.round} miner {self.miner_id}"
        # the keyblock being solved
        self.cur_keyid = None
        self.cur_keyblock: Block = None
        # settings
        self.diffculty = self.background.get_bb_difficulty()
        self.solve_prob_init = self.background.get_solve_prob()
        # self.dmin = self.context.get_dmin() # `warning`:弃用dmin
        self.safe_thre = self.background.get_safe_thre()
        # global state
        self.fathomed_prblms: list[LpPrblm] = []
        self.lower_bound = -sys.maxsize
        self.upper_bound = sys.maxsize
        self.opt_prblms: list[LpPrblm] = []
        self.mbs_unsafe: list[Block] = []
        # the miniblock being solved
        self.pre_block: Block = None
        self.pre_prblm: LpPrblm = None
        self.branching_prblm: LpPrblm = None
        self.prblms_to_branch: list[LpPrblm] = []
        self.solving_pair: SolvingPair = None
        self.solved_pairs: list[tuple[LpPrblm, LpPrblm]] = []
        # 求解miniblock时找到的opt_prblm暂存在这里，中止求解时回滚
        self.optprblm_cache = None
        # the unresolved miniblocks
        self.open_blocks: list[Block] = []
        # 产生keyblock的方式: 'pow'/'withmini'/'pow+withmini'
        self.kb_strategy: str = self.background.get_keyblock_strategy()
        self.withmini_keycache = None
        # searching strategies
        self.opblk_st = self.background.get_openblock_strategy()
        self.opprblm_st = self.background.get_openprblm_strategy()
        self.var_st = VAR_RAND
        # do pow when generating a keyblock
        self.key_pow_target = '000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'
        self.key_pow_nonce = 0
        self.key_pow_hash = None
        self.key_assemble_pre_pname = None

    def update_round(self, round):
        self.round = round
        self.LOG_PREFIX = f"round {self.round} miner {self.miner_id}"

    def get_unpub_num(self, isKey:bool = False, recordSols:bool = False):
        unpubs = []
        unpub_num = len(self.solved_pairs)
        if (not isKey) or (len(self.mbs_unsafe) == 0):
            if not recordSols:
                return unpub_num, unpubs
            for (p1,p2) in self.solved_pairs:
                unpubs.append(p1)
                unpubs.append(p2)
            return unpub_num, unpubs
        # 如果是keyblock发起的，则需要记录unsafe的miniblock
        unpub_num = 0
        for mb in self.mbs_unsafe:
            unpub_num += len(mb.minifield.subprblm_pairs)
        logger.info("%s: unsafe pair num %s, unsafe mbs: %s", self.LOG_PREFIX, 
                    unpub_num, [mb.name for mb in self.mbs_unsafe])
        if not recordSols:
            return unpub_num, unpubs
        for (p1,p2) in self.solved_pairs:
            unpubs.append(p1)
            unpubs.append(p2)
        for mb in self.mbs_unsafe:
            for p in mb:
                unpubs.append(p)
        return unpub_num, unpubs

    def mining_consensus(self, blockchain: Chain, miner_id, isAdversary=None, 
                         input=None, q=None, round=None, prblm_pool=None):
        # TODO: 在回合开始时更新open blocks的状态，而不是取问题时更新
        mineSuccess = False
        if len(self.open_blocks):
            obs = [b.name for b in self.open_blocks]
            logger.info(f"{self.LOG_PREFIX}: open blocks {obs}")
        
        # 没有可求解的keyblock
        if self.cur_keyblock is None:
            self.wait_keyblock()
            return None, mineSuccess

        # 尝试产生key-block
        if self.cur_keyblock.get_fthmstat() or self.background.is_gas_used_up(self.cur_keyblock.get_keyprblm_key().fix_pid):
            newblocks, mineSuccess = self.mining_keyblock(blockchain, miner_id, input, round, q, prblm_pool)
            return newblocks, mineSuccess

        # 若没有正在求解的问题对，尝试获取新问题来branch
        if self.solving_pair is None:
            if len(self.prblms_to_branch) == 0:
                # 尝试从open blocks中获取新的问题
                if len(self.open_blocks) == 0 and not self.cur_keyblock.get_fthmstat():
                    self.wait_openblock()
                    return None, mineSuccess
                if not self.get_prblm_from_open_blocks():
                    wmsg = f"{self.LOG_PREFIX}: get problem error! "
                    logger.warning(wmsg)
                    return None, mineSuccess
            # 获取问题并branch出子问题对
            self.get_prblm_and_branch()

        # 尝试产生mini-block
        newblocks, mineSuccess = self.mining_miniblock(miner_id, input, round)
        if mineSuccess:
            return newblocks, mineSuccess
        return None, mineSuccess


    def wait_keyblock(self):
        logger.info(f"{self.LOG_PREFIX}: waiting for further keyblock")


    def wait_openblock(self):
        logger.info(f"{self.LOG_PREFIX}: waiting for open blocks")


    def mining_miniblock(self, miner_id, input, round):
        """
        挖取miniblock
        """
        mining_success = False
        # continue to solve current sub-problem pair
        solve_finished = self.solving_pair.try_solve_cur_prblms(round)
        if not solve_finished:
            return None, mining_success
        # if current sub-problem pair is sloved, check fathomed, prune, update the bound
        p1 = self.solving_pair.p1
        p2 = self.solving_pair.p2
        self.prblmpair_assemble(p1, p2)
        # whether satisfy generating miniblock rule
        meetGenRule = self.check_mb_gen_rule(p1, p2)
        self.solving_pair = None
        if not meetGenRule:
            return None, mining_success
        # assemble a miniblock
        mined_mb = self.miniblock_assemble(self.pre_block, miner_id, input, round)
        if mined_mb is None:
            return None, mining_success
        self.solved_pairs = []
        mining_success = True
        self.adopt_lowerbound_cache()
        # do some checks before publish
        self.check_subtree_depth(mined_mb)
        # 如果是要和keyblock一起发布的，则组成keyblock
        if (self.kb_strategy != 'pow' and self.withmini_keycache is not None):
            newb = self.set_publish_kb(self.withmini_keycache, mined_mb)
            self.withmini_keycache = None
            return newb, mining_success
        # 安全性检查，不满足则存至miniblocks_unsafe
        self.check_mb_safety(mined_mb)
        newb = self.set_publish_mb(mined_mb)
        imsg = (f"{self.LOG_PREFIX}: mined a miniblock {newb.miniblock.name} "
                f"containing {newb.miniblock.get_subpair_names_mini()}")
        logger.info(imsg)
        return newb, mining_success

    def check_mb_gen_rule(self, cur_p1: LpPrblm, cur_p2: LpPrblm):
        root_pheight = self.solved_pairs[0][0].pheight
        atk_theroy = 1
        # 计算现有的攻击成功概率
        for (p1, _) in self.solved_pairs:
            atk_theroy *= 1 / p1.pre_rest_x
        p_dep = cur_p1.pheight - root_pheight + 1
        logger.info(f"{self.LOG_PREFIX}: checking mb gen-rule curprblm "
                    f"{cur_p1.pname}, atk_theroy: {atk_theroy}, depth:{p_dep}")
        # 检查是否满足难度值要求并且满足安全性要求
        if (cur_p1.pheight - root_pheight + 1 < self.diffculty
                or atk_theroy > self.safe_thre):
            if not cur_p1.fathomed:
                self.prblms_to_branch.append(cur_p1)
            if not cur_p2.fathomed:
                self.prblms_to_branch.append(cur_p2)
        if len(self.prblms_to_branch) != 0:
            return False
        logger.info(f"{self.LOG_PREFIX}: meet mb gen-rule")
        return True


    def prblmpair_assemble(self, p1: LpPrblm, p2: LpPrblm):
        """
        当前子问题对求解完毕, 检查它们的fathomed状态, 更新bound，并写入
        """
        infeas1, wrs1, int_soln1, p1.fathomed = self.check_fathomed(p1)
        infeas2, wrs2, int_soln2, p2.fathomed = self.check_fathomed(p2)
        self.prune_and_update_bound(p1, infeas1, wrs1, int_soln1, 'cache')
        self.prune_and_update_bound(p2, infeas2, wrs2, int_soln2, 'cache')
        p1.fthmd_state = p1.fathomed
        p2.fthmd_state = p2.fathomed
        if wrs1:
            if self.optprblm_cache is not None:
                p1.lb_prblm = self.optprblm_cache
            elif len(self.opt_prblms) > 0:
                p1.lb_prblm = random.choice(self.opt_prblms)
        if wrs2:
            if self.optprblm_cache is not None:
                p2.lb_prblm = self.optprblm_cache
            elif len(self.opt_prblms) > 0:
                p2.lb_prblm = random.choice(self.opt_prblms)
        if self.evaluation.recordGasSolErrs:
            sol = p1.z_lp if int_soln1 and not wrs1 else None
            sol = p2.z_lp if int_soln2 and not wrs2 else sol
            ix = p1.x_lp if int_soln1 and not wrs1 else None
            ix = p2.x_lp if int_soln2 and not wrs2 else ix
            if sol is not None: 
                consume_gas = self.background.get_total_gas()-self.background.get_rest_gas(self.cur_keyblock.get_keyprblm_key().fix_pid)
                self.evaluation.get_round_gas_solution(self.round, consume_gas, sol, self.cur_keyblock.get_keyprblm_key().iz_pulp, ix)
        self.solved_pairs.append((p1, p2))


    def mining_keyblock(self, blockchain, miner_id, input, round, q, prblm_pool):
        """
        尝试产生keyblock, 
        如果现在求解的keyblock的fathomed_state为True, 做PoW, 以产生keyblock
        """
        logger.info(f"{self.LOG_PREFIX}: trying to mine a keyblock")
        mined_kb, mining_success = self.keyblock_assemble(
            blockchain, miner_id, input, round, q, prblm_pool)
        if mining_success:
            key_pname = (mined_kb.keyfield.key_tx.data.pname
                         if mined_kb.keyfield.key_tx is not None else None)
            tx_nonce = (mined_kb.keyfield.key_tx.tx_nonce
                        if mined_kb.keyfield.key_tx is not None else None)
            logger.info(f"{self.LOG_PREFIX}:"
                        f" mined a keyblock{mined_kb.name}"
                        f" containing {key_pname} with tx {tx_nonce}")
            # 如果采用pow产生keyblock，直接发布
            if self.kb_strategy == 'pow':
                newblock = self.set_publish_kb(mined_kb)
                self.switch_key(mined_kb, "inner")
                return newblock, mining_success
            # 不仅仅采用pow，需要接着产生一个miniblock
            self.withmini_keycache = mined_kb
            self.switch_key(mined_kb, "inner")
            return None, False
        return None, mining_success


    def set_publish_mb(self, new_miniblock: Block):
        """
        将待发布的miniblock组合为`NewBlock`
        """
        if new_miniblock.iskeyblock:
            return None
        publishing_block = NewBlocks(False, new_miniblock, None, [], None)
        return publishing_block

 
    def set_publish_kb(self, new_keyblock: Block, mb_with_kb: Block = None):
        """
        将待发布的keyblock与附带的其他miniblock组合为`NewBlock`
        """
        if not new_keyblock.iskeyblock:
            return None
        newblock = NewBlocks(None, None, None, [], None)
        newblock.iskeyblock = True
        newblock.keyblock = new_keyblock
        newblock.mbs_unsafe = [mnb for mnb in self.mbs_unsafe]
        if mb_with_kb is not None:
            newblock.mb_with_kb = mb_with_kb
        # 如果需要连带miniblock一起，此时才清理`self.miniblocks_unsafe`
        if self.kb_strategy != "pow":
            self.mbs_unsafe = []
        logger.info(f"{self.LOG_PREFIX}: publish a keyblock "
                    f"{newblock.keyblock.name} with unsafe blocks "
                    f"{[mnb.name for mnb in newblock.mbs_unsafe]}")
        return newblock


    def cal_attack_rate_pref(self, block: Block):
        """
        计算攻击成功理论值，简化版
        """
        attack_rate = 1
        for (p1, _) in block.minifield:
            attack_rate *= (1 / p1.pre_rest_x)
        return attack_rate


    def check_mb_safety(self, block: Block):
        """
        检查miniblock是否达到安全指标要求
        return: bool True:达到安全性要求
        """
        if block.minifield.atk_rate == 0:
            # 如果未计算过，计算理论的被攻击概率
            attack_rate = self.cal_attack_rate_pref(block)
            block.minifield.atk_rate = attack_rate
            logger.info(f"{self.LOG_PREFIX}: check again block "
                        f"{block.name} atk_theory: {attack_rate}")
        if block.minifield.atk_rate > self.safe_thre:
            # 不满足安全性要求，保存至miniblocks_unsafe
            if block not in self.mbs_unsafe:
                self.mbs_unsafe.append(block)
                mbuns = [b.name for b in self.mbs_unsafe] if self.mbs_unsafe else []
                logger.info(f"{self.LOG_PREFIX}: Block {block.name} not meet "
                            f"safe_thre {self.safe_thre}, cur unsafe mbs: {mbuns}")
            return False
        return True


    def check_subtree_depth(self, block: Block):
        """
        检查miniblock sub-tree深度是否满足diffculty/dmin要求
        """
        """
        `warning`:弃用dmin
        if prblm_layer_num < self.dmin:
            # 小于dmin，保存至miniblocks_unsafe
            self.miniblocks_unsafe.append(block)
        """
        prblm_layer_num = block.get_solve_tree_depth()
        if prblm_layer_num >= self.diffculty:
            # 小于难度，必然求解完毕，不存至open_blocks
            self.open_blocks.append(block)
            # self.open_blocks.insert(0, block)


    def get_prblm_from_open_blocks(self):
        """
        从open_blocks中选择一个子问题, 以此为父问题求解一个miniblock
        """
        # get a block
        getSuccess = False
        open_block = None
        possib_prblms = []
        if self.opprblm_st != OP_BEST:
            while len(possib_prblms) == 0 and len(self.open_blocks) > 0:
                open_block, possib_prblms = self.get_open_block_and_prblms()
        else:
            open_block, possib_prblms = self.get_open_prblm_bestbound()

        if open_block is None:
            obs = [b.name for b in self.open_blocks] if len(self.open_blocks) else []
            wmsg = f"{self.LOG_PREFIX}: failed to get a open block within {obs}"
            logger.warning(wmsg)
            return getSuccess

        self.pre_block = open_block
        logger.info(f"{self.LOG_PREFIX}: get open block {open_block.name} with "
                    f"possible prblms {[p.pname for p in possib_prblms]}")
        if len(possib_prblms) > 0:
            prblm = self.select_branch_prblm(possib_prblms)
            self.pre_prblm = prblm
            self.prblms_to_branch.append(prblm)
            getSuccess = True
        # if the block is not solved yet, put it back
        if len(possib_prblms) > 1:
            # self.open_blocks.append(self.pre_block)
            self.open_blocks.insert(0, self.pre_block)
        if not getSuccess:
            logger.warning(f"{self.LOG_PREFIX}: failed to branch "
                           f"a problem from open block {open_block.name}")
        return getSuccess


    def select_branch_prblm(self, open_prblms: list[LpPrblm]):
        """
        select a problem  to branch from the possib_prblms
        and append it to the `self.prblms_to_solve`
        """
        if len(open_prblms) == 0:
            raise ValueError("no open problems to select")
        if self.opprblm_st == OP_SPEC:
            prblm = open_prblms[0]
        elif self.opprblm_st == OP_RAND:
            prblm = random.choice(open_prblms)
        elif self.opprblm_st == OP_BEST:
            if len(open_prblms) != 1:
                raise ValueError("OP_BEST get problem error!")
            prblm = open_prblms[0]
        logger.info(f"{self.LOG_PREFIX}: select problem {prblm.pname} to branch")
        return prblm


    def get_open_block_and_prblms(self):
        open_block = self.get_open_block()
        if open_block is None:
            return None, []
        possib_prblms = self.get_open_prblms_from_block_and_update_fstate(open_block)
        if not open_block.iskeyblock and open_block.get_fthmstat():
            open_block.update_solve_tree_fthmd_state_mini()
            self.reorg_open_blocks()
            return None, []
        return open_block, possib_prblms
    

    def get_open_prblm_deepfirst(self):
        """
        从所有open block中的open problems中找到最深的问题
        : return open_block 选中的区块
        : return open_prblm 选中的子问题
        """
        if len(self.open_blocks) <= 0:
            return None
        ob_idxs = []
        dep_ops = []
        maxdep_op = 0
        for i, ob in enumerate(self.open_blocks):
            # 获取open problems（同时更新问题的fathomed状态）
            ops = self.get_open_prblms_from_block_and_update_fstate(ob)
            # 更新solve tree状态
            if not ob.iskeyblock and ob.get_fthmstat():
                ob.update_solve_tree_fthmd_state_mini()
                self.reorg_open_blocks()
                continue
            # 获取最深的问题列表
            for p in ops:
                if p.pheight > maxdep_op:
                    ob_idxs.clear()
                    maxdep_op = p.pheight
                    ob_idxs.append(i)
                    dep_ops.clear()
                    dep_ops.append(p)
                elif p.pheight == maxdep_op:
                    ob_idxs.append(i)
                    dep_ops.append(p)
        # 随机选择一个最深的问题，同时返回对应所在的block
        open_prblm = random.choice(dep_ops)
        open_block = self.open_blocks.pop(ob_idxs[dep_ops.index(open_prblm)])
        return open_block, open_prblm

    def get_open_prblm_bestbound(self):
        """
        从所有open block中的open problems中找到最优的问题，即松弛解最大的问题
        : return open_block 选中的区块
        : return open_prblm 选中的子问题
        """
        if len(self.open_blocks) <= 0:
            return None, None
        ob_idxs = []
        ob_pnums = []
        best_ops:list[LpPrblm] = []
        best_bd = -sys.maxsize
        for i, ob in enumerate(self.open_blocks):
            # 获取open problems（同时更新问题的fathomed状态）
            ops = self.get_open_prblms_from_block_and_update_fstate(ob)
            # 更新solve tree状态
            if (not ob.iskeyblock) and ob.minifield.bfthmd_state:
                ob.update_solve_tree_fthmd_state_mini()
                continue
            # 获取最优的问题列表
            for p in ops:
                if p.z_lp > best_bd:
                    ob_idxs.clear()
                    ob_pnums.clear()
                    best_bd = p.z_lp
                    ob_idxs.append(i)
                    ob_pnums.append(len(ops))
                    best_ops.clear()
                    best_ops.append(p)
                elif p.z_lp == best_bd:
                    ob_idxs.append(i)
                    best_ops.append(p)
                    ob_pnums.append(len(ops))
        # 随机选择一个最优的问题，同时返回对应所在的block
        if len(best_ops)==0:
            return None, None
        open_prblm = random.choice(best_ops)
        pop_idx = next((i for i, p in enumerate(best_ops) 
                        if p.pname == open_prblm.pname), None)
        open_block = self.open_blocks.pop(ob_idxs[pop_idx])
        if ob_pnums[best_ops.index(open_prblm)] > 1:
            self.open_blocks.insert(0, open_block)
        logger.info("%s: best ob: get open block %s and open prblm %s, "
                    "lowest_ops %s, pop_idx %s, ob_idxs %s, ob_pnums %s", 
                    self.LOG_PREFIX, open_block.name, open_prblm.pname, 
                    [p.pname for p in best_ops], pop_idx, ob_idxs, ob_pnums)
        self.reorg_open_blocks()
        return open_block, [open_prblm]

    def get_open_block(self):
        """
        从open_blocks获取一个open_block
        """
        if len(self.open_blocks) <= 0:
            return None
        if self.opblk_st == OB_SPEC:
            sorted(self.open_blocks, key=lambda x: x.name, reverse=True)
            open_block = self.open_blocks.pop(0)
        elif self.opblk_st == OB_RAND:
            open_block = random.choice(self.open_blocks)
            self.open_blocks.remove(open_block)
        elif self.opblk_st == OB_DEEP:
            open_block = self.get_open_block_deepfirst()
        elif self.opblk_st == OB_BREATH:
            open_block = self.get_open_block_breathfirst()
        return open_block

    def get_open_block_deepfirst(self):
        """
        open block选取原则：深度优先
        """
        if len(self.open_blocks) <= 0:
            return None
        dep_idxs = []
        maxdep = 0
        for i, b in enumerate(self.open_blocks):
            bheight = b.get_height()
            if bheight > maxdep:
                dep_idxs.clear()
                dep_idxs.append(i)
                maxdep = bheight
            elif bheight == maxdep:
                dep_idxs.append(i)
        idx = random.choice(dep_idxs)
        open_block = self.open_blocks.pop(idx)
        return open_block

    def get_open_block_breathfirst(self):
        """
        open_block选取原则：广度优先
        """
        if len(self.open_blocks) <= 0:
            return None
        dep_idxs = []
        mindep = sys.maxsize
        for i, b in enumerate(self.open_blocks):
            bheight = b.get_height()
            if bheight < mindep:
                dep_idxs.clear()
                dep_idxs.append(i)
                mindep = bheight
            elif bheight == mindep:
                dep_idxs.append(i)
        idx = random.choice(dep_idxs)
        open_block = self.open_blocks.pop(idx)
        return open_block


    def get_open_prblms_from_block_and_update_fstate(self, cur_block: Block):
        """
        Get a unsolved problems form a open block. 
        同时也更新curblock的fathomed state
        return: unsolved_prblms(list[LpPrblm])
        """
        unfthmd_prblms = cur_block.get_unfthmd_leafps()
        # 获取未被求解的问题
        solved_pnames = []
        if len(cur_block.next) > 0:
            for next_b in cur_block.next:
                if next_b.iskeyblock is False:
                    solved_pnames.append(next_b.minifield.pre_pname)
        unsolved_prblms = [p for p in unfthmd_prblms if p.pname not in solved_pnames]
        # 获取未被求解并且unfathomed的问题，即open problems
        open_prblms:list[LpPrblm] = []
        for p in unsolved_prblms:
            _, _, _, fthmd = self.check_fathomed(p)
            if not fthmd:
                open_prblms.append(p)
                continue
            # 如果该问题fathomed，更新其状态
            if not p.fthmd_state:
                p.fthmd_state = fthmd
                if cur_block.iskeyblock:
                    return []
                cur_block.minifield.update_fathomed_state()
        return open_prblms
    
    def check_again_mb_fthmd_state(self, block:Block):
        """检查并更新Miniblock的fthmd_state"""
        if block.iskeyblock:
            raise ValueError(f"update_miniblock_fthmd_state: {block.name} is a keyblock")
        unfthmd_prblms = block.get_unfthmd_leafps()
        # updateFstat = False
        for p in unfthmd_prblms:
            _, wrs, _, fthmd = self.check_fathomed(p)
            if fthmd and not p.fthmd_state:
                # updateFstat = True
                p.fthmd_state = fthmd
                p.fathomed = fthmd
                if wrs and self.optprblm_cache is not None:
                    p.lb_prblm = self.optprblm_cache
                elif wrs and self.optprblm_cache is None and len(self.opt_prblms) > 0 :
                    p.lb_prblm = random.choice(self.opt_prblms)
        block.minifield.update_fathomed_state()
        return block.get_fthmstat()

    def get_prblm_and_branch(self):
        """
        Get a problem from the `prblms_to_branch` and branch it
        """
        self.branching_prblm = self.prblms_to_branch.pop(0)
        idx = self.select_branch_var(self.branching_prblm)
        new_p1, new_p2 = self.branch(self.branching_prblm, idx)
        # ready to solve the pair
        key_lp = self.cur_keyblock.get_keyprblm_key()
        self.solving_pair = SolvingPair(new_p1, new_p2, key_lp, self.solve_prob_init)


    def select_branch_var(self, lp_prblm: LpPrblm):
        """
        Select a varible to branch -- random
        """
        nonint_idx = [idx for idx, x in enumerate(lp_prblm.x_lp)
                      if not x.is_integer() and idx not in lp_prblm.conti_vars]
        if self.var_st == VAR_RAND:
            idx = random.choice(nonint_idx)
        elif self.var_st == VAR_SPEC:
            idx = nonint_idx[-1]
        else:
            raise ValueError("Var select strategy Error!")
        return idx


    def reorg_open_blocks(self):
        """
        删除unresolved_blocks中连接到状态为fathomed的区块,
        如果正在求解的区块也连接到了状态为fathomed的区块, 终止求解
        """
        # reorg unresolved blocks
        del_blocks = []
        for block in self.open_blocks:
            if not block.iskeyblock:
                can_del = block.check_link2_fthmd_prblm()
                if can_del:
                    del_blocks.append(block)

        if len(del_blocks) > 0:
            self.open_blocks = [b for b in self.open_blocks
                                if b not in del_blocks]
        # handle the solving prblm
        if self.pre_prblm:
            if self.pre_prblm.fthmd_state:
                self.cancel_solving_prblm()
            elif (not self.pre_block.iskeyblock and
                  self.pre_block.check_link2_fthmd_prblm()):
                self.cancel_solving_prblm()


    def setparam(self, target):
        pass


    def valid_chain(self, lastblock: Block):
        return True


    def valid_block(self, block_chain: Chain, block: Block):
        if block.iskeyblock:
            return self.valid_keyblock(block)
        # miniblock
        soln_valid = True
        fthom_valid = True
        for (p1, p2) in block.minifield:
            # validate the solution
            soln_valid = (
                soln_valid
                and self.valid_solution(p1)  
                and self.valid_solution(p2))
            # validate fathomed
            fthom_valid = (
                fthom_valid
                and self.valid_fathomed(block_chain, block, p1)
                and self.valid_fathomed(block_chain, block, p2))
        # validate the diffculty
        difficulty_valid = self.valid_difficulty(block)
        return soln_valid and fthom_valid and difficulty_valid

    def valid_keyblock(self, keyblock:Block):
        if keyblock.keyfield.key_tx is not None:
            self.prune_and_update_bound(keyblock.get_keyprblm_key(), updateLBCrtl = False)
            return True
        return True
    

    def valid_solution(self, lpprblm: LpPrblm):
        return True


    def valid_fathomed(self, block_chain: Chain, cur_block: Block, lpprblm: LpPrblm):
        '''
        Validate if the sub-problem is fathomed as it claimed.
        If the validation result is True, update the bound
        '''
        fthmdValid = False
        # 以lb_base为基准检查fathomed
        lb_base =  sys.maxsize if lpprblm.lb_prblm is None else lpprblm.lb_prblm.z_lp 
        infeas, wrs, int_soln, prblm_fthmd = self.check_fathomed(lpprblm, lb_base)

        if lpprblm.fathomed:
            # 该问题被声明fathomed
            fthmdValid = self.valid_claim_fathomed(
                block_chain, cur_block, lpprblm, prblm_fthmd, wrs)
        else:  # 该问题未被声明为fathomed
            fthmdValid = True
            if prblm_fthmd:
                logger.warning(
                    f"{self.LOG_PREFIX}: The problem {lpprblm.pname} is fathomed"
                    f"(infeas:{infeas}, wrs:{wrs}, int_soln:{int_soln}) but claimed not")
        # 如果验证通过，更新下界
        keyprblm_id = lpprblm.pname[0][0]
        if fthmdValid and keyprblm_id == self.cur_keyid:
            self.prune_and_update_bound(lpprblm, infeas, wrs, int_soln, updateLBCrtl = False)
        return fthmdValid


    def valid_claim_fathomed(self, block_chain: Chain, cur_block: Block, lpprblm: LpPrblm,
                             prblm_fthmd: bool, wrs: bool):
        """
        该问题被声明为fathomed, 如果是infeas或int_soln, 验证通过；
        如果是wrs, 还需要检查提供的指定lowerbound的问题是否存在
        """
        return True
        fthmd_vali = False
        # 未检查出fathomed，不通过
        if not prblm_fthmd:
            wmsg = (f"{self.LOG_PREFIX}: The problem {lpprblm.pname} "
                    "is not fathomed but claimed fathomed")
            logger.warning(wmsg)
            return fthmd_vali
        # 检查出infeas或int_soln
        if prblm_fthmd and not wrs:
            fthmd_vali = True
            return fthmd_vali
        # 检查出wrs且给出下界的问题本身包含在区块里
        if lpprblm.lb_prblm is None:
            fthmd_vali = True if lpprblm.z_lp > self.cur_keyblock.get_keyprblm_key().init_iz else False
            return fthmd_vali
        # 没有的话从链里面找
        vali_keyid = lpprblm.pname[0][0]
        vali_keyhead = None
        if self.cur_keyblock is not None and vali_keyid == self.cur_keyid:
            vali_keyhead = self.cur_keyblock
        else:
            local_kbs = block_chain.get_keyblocks_pref()
            for kb in local_kbs:
                if kb.keyfield.key_tx is None:
                    continue
                if vali_keyid == kb.keyfield.key_tx.data.pname[0][0]:
                    vali_keyhead = kb
                    break
        if vali_keyhead is None:
            wsmg = (f"{self.LOG_PREFIX}: validating miniblock{cur_block.name}: "
                    f"Not found the keyblock with keyid{vali_keyid} in local chain")
            logger.warning(wsmg)
            return fthmd_vali
        # q = [self.cur_keyblock]
        q = [vali_keyhead]
        while q:
            block = q.pop(0)
            if lpprblm.lb_prblm in block:
                fthmd_vali = True
                return fthmd_vali
            q.extend([b for b in block.next if not b.iskeyblock])
        # 没有找到给出下界的问题，不通过
        if not fthmd_vali:
            logger.warning(f"{self.LOG_PREFIX}: "
                           f"The problem {lpprblm.lb_prblm.pname} yielding the "
                           f"lowerbound of {lpprblm.pname} is not found!")
            return fthmd_vali


    def valid_difficulty(self, block: Block):
        return True


    def miniblock_assemble(self, pre_block: Block, miner_id, input, round):
        '''
        If all the subproblems are solved, generate a miniblock.
        :param pre_block: The block which the miniblock links to
        :param miner_id: The id of the miner 
        '''
        if pre_block is None:
            return None
        height = pre_block.blockhead.height
        block_name = f'B{str(self.background.get_block_number())}'
        prehash = pre_block.name  # 姑且用blockname代替block hash
        blockhash = block_name
        mb_head = BlockHead(prehash, blockhash, height + 1, miner_id, round)
        new_mb = Block(block_name,
                       iskeyblock=False,
                       blockhead=mb_head,
                       content=input,
                       blocksize_MB=self.background.get_blocksize())
        new_mb.set_minifield(self.pre_prblm, self.solved_pairs)
        self.check_again_mb_fthmd_state(new_mb)
        logger.info("%s: %s fathomed state %s", self.LOG_PREFIX, 
                    new_mb.name, new_mb.get_fthmstat())
        return new_mb


    def keyblock_assemble(self, chain: Chain, miner_id, input,
                          round, key_pow_q, prblm_pool):
        '''Try to generate a keyblock, do key PoW
        and add a new problem from the prblm_pool'''
        genKeyblockSuccess = False
        preKeyFeasible = True
        powSuccess = False if self.key_pow_hash is None else True
        keyblock = None
        pre_pname = self.get_key_pre_pname(chain)
        if ((self.kb_strategy == 'pow' or 
             self.kb_strategy == 'pow+withmini') and not powSuccess):
            logger.info("%s: doing PoW, key fathomed state %s", 
                        self.LOG_PREFIX, self.cur_keyblock.get_fthmstat())
            powSuccess = self.keyblock_pow(miner_id, str(pre_pname), key_pow_q)
        elif self.kb_strategy == 'withmini':
            powSuccess = True
        if powSuccess:
            consume_gas = self.background.get_total_gas()-self.background.get_rest_gas(self.cur_keyblock.get_keyprblm_key().fix_pid)
            self.evaluation.get_round_gas_solution(self.round, consume_gas, self.upper_bound, self.cur_keyblock.get_keyprblm_key().iz_pulp)
            # 产生一个keyblock
            block_name = f'B{str(self.background.get_block_number())}'
            prehash = chain.lastblock.name
            blockhash = block_name
            height = chain.lastblock.blockhead.height + 1  
            blockhead = BlockHead(prehash, blockhash, height, miner_id, round)
            keyblock = Block(block_name, True, blockhead, content=input, 
                             blocksize_MB=self.background.get_blocksize())
            # get a prblm from the prblm pool, and assemble a keyblock
            if len(self.opt_prblms) == 0:
                preKeyFeasible = False
            key_tx, solveSuccess = self.get_and_slove_next_keytx(prblm_pool, pre_pname)
            if not solveSuccess:
                return None, genKeyblockSuccess
            accect_mbs = self.get_fathomed_prblms_by_chain()
            logger.info(f"{self.LOG_PREFIX}: opt_prblms: {[p.pname for p in self.opt_prblms]}, opt_z:{[p.z_lp for p in self.opt_prblms]} "
                        f"optprblm_cache:{self.optprblm_cache.z_lp if self.optprblm_cache is not None else None}")
            key_height = self.cur_keyblock.keyfield.key_height + 1
            keyblock.set_keyfield(
                self.key_pow_hash, self.key_pow_nonce, key_height,
                self.cur_keyblock, pre_pname, preKeyFeasible, self.opt_prblms,
                self.fathomed_prblms, key_tx, accect_mbs, self.cur_keyblock.get_keyprblm_key().iz_pulp)
            self.key_pow_nonce = 0
            self.key_pow_hash = None
            self.key_assemble_pre_pname = None
            prekey = (keyblock.keyfield.pre_kb.name 
                      if keyblock.keyfield.pre_kb else None)
            logger.info(f"{self.LOG_PREFIX}: mined a keyblock {block_name}, "
                        f"key_height{key_height}, pre_keyblock{prekey}")
            genKeyblockSuccess = True
        return keyblock, genKeyblockSuccess


    def get_fathomed_prblms(self):
        """
        获取接受链上所有的fathomed_problems
        """
        q = [self.cur_keyblock]
        while q:
            block = q.pop(0)
            for p in block:
                if p.fathomed:
                    self.fathomed_prblms.append(p)
            if len(block.next) == 0:
                continue
            next_mbs = []
            for next_b in block.next:
                if not next_b.iskeyblock:
                    if next_b.minifield.bfthmd_state:
                        next_mbs.append(next_b)
            q.extend(next_mbs)


    def get_fathomed_prblms_by_chain(self):
        """
        利用chain中的函数 `get_acpmbs_after_kb`,
        获取接受链上所有的fathomed_problems，并返回选择的接收链
        """
        acpmbs = self.local_chain.get_acpmbs_after_kb_and_label_forks(self.cur_keyblock)
        if len(acpmbs) == 0:
            return []
        for mb in acpmbs:
            for p in mb:
                if p.fathomed:
                    self.fathomed_prblms.append(p)
        return [mb.name for mb in acpmbs]


    def keyblock_pow(self, miner_id, input, key_pow_q):
        '''
        产生keyblock时计算pow
        '''
        # self.doing_key_pow = True
        pow_success = False
        key_hash = None
        for _ in range(key_pow_q):
            self.key_pow_nonce += 1
            key_hash = hashsha256([miner_id, self.key_pow_nonce, input])
            if int(key_hash, 16) < int(self.key_pow_target, 16):
                pow_success = True
                self.key_pow_hash = key_hash
                break
        return pow_success

    def get_key_pre_pname(self, chain: Chain):
        if self.key_assemble_pre_pname is not None:
            return self.key_assemble_pre_pname
        if len(self.opt_prblms) > 0:
            # # 如果上个问题有整数解，防止该整数解没有被加到链上
            # # print([p.pname for p in self.opt_prblms])
            # mbs_for_prekey = chain.get_mbs_after_kb(self.cur_keyblock)
            # # print([b.name for b in mbs_for_prekey])
            # int_ps_for_prekey = []
            # for mb in mbs_for_prekey:
            #     for (p1, p2) in mb.minifield:
            #         if p1.all_integer():
            #             int_ps_for_prekey.append(p1)
            #         if p2.all_integer():
            #             int_ps_for_prekey.append(p2)
            # # print([(p.pname, p.z_lp) for p in int_ps_for_prekey])
            # opt_prblms = [p for p in self.opt_prblms if p in int_ps_for_prekey]
            # # print([(p.pname, p.z_lp) for p in self.opt_prblms])
            pre_pname = random.choice(self.opt_prblms).pname
        else:
            # 如果没有整数解的话, 随机选择末端miniblock中最深的问题
            deepest_subprblms = chain.lastblock.get_deepest_subprblms()
            pre_pname = random.choice(deepest_subprblms).pname
        self.key_assemble_pre_pname = pre_pname
        return pre_pname

    def get_and_slove_next_keytx(self, prblm_pool: TxPool, pre_pname:tuple):
        """
        从prblm_pool中取一个原始问题tx, 以加入keyfield
        """
        if len(prblm_pool.pending) <= 0:
            key_tx = None
            print(f'{self.LOG_PREFIX}: No more problems to solve in the pool.')
            return key_tx, True

        key_tx = prblm_pool.pending[0]
        keyprblm = key_tx.data
        
        solveSuccess = lpprblm.solve_lp(keyprblm, self.solve_prob_init)
        if not solveSuccess:
            logger.info(f"{self.LOG_PREFIX}: solving next key-problem failed")
            return None, solveSuccess
        
        logger.info(f"{self.LOG_PREFIX}: solving next key-problem success")
        prblm_pool.pending.pop(0)
        prblm_pool.reorg()
        keyprblm.pname = ((self.background.key_id_generator(), 0), 0)
        keyprblm.pre_pname = pre_pname
        # lpprblm.solve_ilp_by_pulp(keyprblm)
        _, _, _, keyprblm.fathomed = self.check_fathomed(keyprblm, sys.maxsize)
        keyprblm.fthmd_state = keyprblm.fathomed
        keyprblm.timestamp = self.round
        return key_tx, solveSuccess


    def switch_key(self, keyblock: Block, kb_from="outer", round=None):
        """
        切换到新的keyblock进行求解, 相当于切换到新的状态，需要清理掉旧状态的信息，
        但有些信息如`self.miniblocks_unsafe`需要视情况保留。
        :param keyblock: 将要切换到的keyblock
        :param kb_from: 该keyblock的来源，"outer"：从外界接收的，"inner"：自己产生的
        """
        round = self.round if round is None else round
        logger.info(f"round{round} miner{self.miner_id}: "
                    f"switch to a new keyblock {keyblock.name}")
        self.cancel_solving_prblm()
        self.key_pow_nonce = 0
        self.key_pow_hash = None
        self.key_assemble_pre_pname = None
        self.prblms_to_branch = []
        self.solving_pair = None
        self.solved_pairs = []
        self.open_blocks = []
        self.fathomed_prblms = []
        self.opt_prblms = []
        self.lower_bound = -sys.maxsize
        self.upper_bound = sys.maxsize
        self.optprblm_cache = None
        # 来源于外界的keyblock，需要清理全部的旧信息
        if kb_from == "outer":
            self.mbs_unsafe = []
        else:
            # 如果keyblock仅仅使用pow，此时就清理
            if self.kb_strategy == 'pow':
                self.mbs_unsafe = []
        # if self.doing_key_pow:
        #     self.doing_key_pow = False
        if keyblock.iskeyblock is False:
            raise ValueError('The block is not a keyblock !')
        if keyblock.keyfield.key_tx is not None:
            self.cur_keyblock = keyblock
            kp = keyblock.keyfield.key_tx.data
            self.cur_keyid = kp.pname[0][0]
            if len(keyblock.next) == 0:
                self.open_blocks.append(keyblock)
            if kp.init_iz is not None:
                self.upper_bound = kp.init_iz
        else:
            self.cur_keyid = -1
            self.cur_keyblock = None


    def cancel_solving_prblm(self):
        """
        终止正在解决的问题，
        使用条件: 接收到的区块的父问题与当前求解的miniblock父问题相同
        """
        self.pre_block = None
        self.branching_prblm = None
        self.pre_prblm = None
        self.solved_pairs = []
        self.prblms_to_branch = []
        self.solving_pair = None
        self.optprblm_cache = None
        # self.opt_prblm = None


    def check_all_branches_fathomed(self):
        """
        Check whether all branches are fathomed. If not, return False,
        and append the block with unsolved or unftmd_prblm
        """
        allBranchesFthmd = True
        q: list[Block] = [self.cur_keyblock]
        while q:
            block = q.pop(0)
            unsolved_unftmd_prblms = self.get_open_prblms_from_block_and_update_fstate(block)
            if len(unsolved_unftmd_prblms) > 0:
                self.open_blocks.append(block)
                wmsg = (f"{self.LOG_PREFIX}: {block.name} have unsolved and unfathomed "
                        f" subproblem(s): {[p.pname for p in unsolved_unftmd_prblms]}")
                logger.warning(wmsg)
                allBranchesFthmd = False
            else:
                q.extend([b for b in block.next if b.iskeyblock is False])
        return allBranchesFthmd

    
    def branch(self, pre_prblm: LpPrblm, idx):
        '''Branch the given problem and generate two subproblems.'''
        # new constraints
        inc_constr1 = copy.deepcopy(pre_prblm.inc_constrs)
        inc_constr2 = copy.deepcopy(pre_prblm.inc_constrs)
        inc_constr1.append(IncConstr(idx, -1, -math.ceil(pre_prblm.x_lp[idx])) )
        inc_constr2.append(IncConstr(idx, 1, math.floor(pre_prblm.x_lp[idx])))
        # gen new sub-problems
        p_name0 = (self.cur_keyid, 
                   self.background.autoinc_prblm_id_generator(self.cur_keyid))
        pre_rest_x = len([idx for idx, x in enumerate(pre_prblm.x_lp)
                          if not x.is_integer()])
        key_pname = self.cur_keyblock.get_keypname_key()
        new_p1 = LpPrblm((p_name0, 1), pre_prblm.pname, pre_prblm.pheight + 1, 
                         key_pname=key_pname, x_nk = idx, pre_rest_x = pre_rest_x, 
                         inc_constrs = inc_constr1)
        new_p2 = LpPrblm((p_name0, -1), pre_prblm.pname, pre_prblm.pheight + 1, 
                         key_pname=key_pname, x_nk = idx, pre_rest_x = pre_rest_x, 
                         inc_constrs = inc_constr2)
        new_p1.conti_vars = pre_prblm.conti_vars
        new_p2.conti_vars = pre_prblm.conti_vars
        # lpprblm.solve_lp_by_pulp(new_p1)
        # lpprblm.solve_lp_by_pulp(new_p2)
        return new_p1, new_p2


    def check_fathomed(self, lp_prblm: LpPrblm, upperbound=None):
        '''Judge whether the subproblem branch fathomed.
        return: (infeas_fthmd, wrs_fthmd, int_soln_fthmd, prblm_fthmd)
        :prblm_fthmd = infeas_fthmd or wrs_fthmd or int_soln_fthmd
        '''
        if (lp_prblm.x_lp is None and lp_prblm.z_lp is None
                and lp_prblm.feasible is None):
            raise ValueError(f"{self.LOG_PREFIX}: The problem{lp_prblm.pname} to"
                             "check fathomed is not solved yet")
        # 获取全局上界
        ub = upperbound if upperbound is not None else self.upper_bound
        if self.optprblm_cache is not None:
            ub = ub if ub <= self.optprblm_cache.z_lp else self.optprblm_cache.z_lp

        infeas = False
        wrs = False
        int_soln = False
        # The solution is infeasible.
        if lp_prblm.feasible is False:
            infeas = True
            return infeas, wrs, int_soln, True
        # The objective function is worse than the upperbound.
        if lp_prblm.z_lp > ub:
            wrs = True
            return infeas, wrs, int_soln, True
        # Integer solution and better than the upperbound.
        if lp_prblm.all_integer():
            int_soln = True
            return infeas, wrs, int_soln, True
        return infeas, wrs, int_soln, False


    def prune_and_update_bound(
            self, lp_prblm: LpPrblm, 
            infeas: bool = None,
            wrs: bool = None, 
            int_soln: bool = None,
            opt_save_loc: str = None,
            updateLBCrtl:bool = True):
        """
        :If the solution is integer and better than the lowerbound, 
        the lower_bound is set to z_lp and appended to opt_prblms; 
        :If not fathomed, update the upper_bound.
        :如果预先检查过了fathomed, 直接将检查结果传入.
        :param: opt_save_loc:'cache' or 'opt_prblms'
        :return: update_bound_success
        """
        if infeas is None or wrs is None or int_soln is None:
            infeas, wrs, int_soln, _ = self.check_fathomed(lp_prblm)
        updateLB = False
        updateUB = False
        # update lowerbound
        opt_loc = 'opt_prblms' if opt_save_loc is None else opt_save_loc
        if int_soln:
            self.update_upperbound(opt_loc, lp_prblm)
            # rest_gas = self.background.get_total_gas()-self.background.get_rest_gas(self.cur_keyblock.get_keyprblm_key().fix_pid)
            # 将输出写入文件
            # logger.info(f"{self.LOG_PREFIX}: rest_gas:{rest_gas}; x_lp:{lp_prblm.x_lp}; z_lp:{lp_prblm.z_lp}; "
            #         f"upper_bound:{self.upper_bound}; init_ix:{lp_prblm.init_ix}; init_iz:{lp_prblm.init_iz};\n")
            # self.evaluation.get_round_gas_solution(self.round, rest_gas, lp_prblm.z_lp, self.cur_keyblock.get_keyprblm_key().iz_pulp)
            updateUB = True
        # update lowerbound
        if not infeas and not wrs and updateLBCrtl:
            self.lower_bound = lp_prblm.z_lp
            updateLB = True
        return updateUB, updateLB


    def update_upperbound(self, opt_save_loc: str, lp_prblm: LpPrblm):
        """
        根据opt_save_loc更新下界,
        如果为`cache`就保存在opt_prblm_cache中
        如果为`opt_prblms`就保存在opt_prblms中
        """
        # 如果小于upper_bound，直接更新最优解
        imsg = None
        if lp_prblm.z_lp < self.upper_bound:
            if opt_save_loc == 'opt_prblms':
                self.opt_prblms.clear()
                self.opt_prblms.append(lp_prblm)
                self.upper_bound = lp_prblm.z_lp
                imsg = (f"{self.LOG_PREFIX}: update opt_prblm:"
                        f"{lp_prblm.pname} with z_lp {lp_prblm.z_lp}")
            elif opt_save_loc == 'cache':
                self.optprblm_cache = lp_prblm
                imsg = (f"{self.LOG_PREFIX}: update opt_cache:"
                        f"{lp_prblm.pname} with z_lp {lp_prblm.z_lp}")

        # 如果与upper_bound相等，将其附加到`opt_prblms`
        elif (lp_prblm.z_lp == self.upper_bound and lp_prblm not in self.opt_prblms):
            if opt_save_loc == 'opt_prblms':
                self.opt_prblms.append(lp_prblm)
                imsg = (f"{self.LOG_PREFIX}: append opt_prblm:"
                        f"{lp_prblm.pname} with z_lp {lp_prblm.z_lp}")
            elif opt_save_loc == 'cache':
                self.optprblm_cache = lp_prblm
                imsg = (f"{self.LOG_PREFIX}: append opt_cache:"
                        f"{lp_prblm.pname} with z_lp {lp_prblm.z_lp}")
        if imsg is not None:
            logger.info(imsg)


    def adopt_lowerbound_cache(self):
        """
        如果opt_prblm_cache不为None, 将其更新到opt_prlms中
        """
        if self.optprblm_cache is None:
            return
        if self.optprblm_cache.z_lp < self.upper_bound:
            self.opt_prblms.clear()
            self.opt_prblms.append(self.optprblm_cache)
            self.upper_bound = self.optprblm_cache.z_lp
            imsg = (f"{self.LOG_PREFIX}: adopt opt_cache:"
                    f"{self.optprblm_cache.pname} with z_lp {self.optprblm_cache.z_lp}")
            logger.info(imsg)
        elif (self.optprblm_cache.z_lp == self.upper_bound and
              self.optprblm_cache not in self.opt_prblms):
            self.opt_prblms.append(self.optprblm_cache)
            imsg = (f"{self.LOG_PREFIX}: adopt and append opt_cache:"
                    f"{self.optprblm_cache.pname} with z_lp {self.optprblm_cache.z_lp}")
            logger.info(imsg)
        self.optprblm_cache = None