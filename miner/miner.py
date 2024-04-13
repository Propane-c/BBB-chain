import copy
import logging
import random
import sys
from collections import defaultdict

import data.lpprblm as lpprblm
from data import Block, Chain, NewBlocks,TxPool
from branchbound import BranchBound
from evaluation import Evaluation
from .fork_view import ForkView


RECORD_SOLS = False

logger = logging.getLogger(__name__)


class Miner(object):
    def __init__(self, background, miner_id, q, target, evaluation:Evaluation):
        self.miner_id = miner_id  # 矿工ID
        self.isAdversary = False
        self.q = q
        self.round = 0
        self.LOG_PREFIX = None
        # 链相关
        self.local_chain = Chain(background, self.miner_id)  # 本地维护的区块链
        self.forkviews:list[ForkView] = []
        # 共识相关
        self.consensus = BranchBound(background, self.local_chain, self.miner_id)
        self.consensus.setparam(target)  # 设置共识参数
        # 输入内容相关
        self.input = 0  # 要写入新区块的值
        self.input_tape = []
        # 接收相关
        self.receive_tape: list[NewBlocks] = []
        self.receive_history = []  # 保留最近接收到的3个块
        self.buffer_size = 3
        # 网络相关
        self.neighbor_list = []
        self.processing_delay = 0  # 处理时延
        # txpool
        self.txpool = copy.deepcopy(TxPool())
        # outer evaluation
        self.evaluation = evaluation

    def set_adversary(self, isAdversary: bool):
        """
        设置是否为对手节点
        isAdversary=True为对手节点
        """
        self.isAdversary = isAdversary

    def is_in_local_chain(self, block: Block):
        """
        Check whether a block is in local chain,
        param: block: The block to be checked
        return: flagInLocalChain: Flag whether the block is in local chain."""
        inLocalChain = False
        if block not in self.local_chain:
            inLocalChain = False
        else:
            inLocalChain = True
        return inLocalChain

    def receive_blocks(self, rcvblocks: NewBlocks, round):
        """
        Interface between network and miner.
        Append rcvblock not received before to receive_tape, 
        which will be added to local chain in the next round by `maxvalid`.

        Args:
            rcvblock (Block): The block received from network

        Returns:
            bool: If the rcvblock not in local chain or receive_tape, return True
        """
        
        notReceived = False
        if rcvblocks in self.receive_tape:
            return notReceived
        if not rcvblocks.iskeyblock:
            notReceived = self.receive_miniblock(rcvblocks)
            fthmd = rcvblocks.miniblock.get_fthmstat()
            logger.info("%s: receives a miniblock %s, success: %s, fathomed_state:%s",
                        self.LOG_PREFIX, rcvblocks.miniblock.name, notReceived, fthmd)
        elif rcvblocks.iskeyblock:
            notReceived = self.receive_keyblock(rcvblocks)
            key_id = rcvblocks.keyblock.get_keyid()
            logger.info("%s: receives a keyblock %s with keyid %s, success: %s", 
                        self.LOG_PREFIX, rcvblocks.keyblock.name, key_id, notReceived)
        else:
            err = (f"Miner{self.miner_id}: The rceived block {rcvblocks.miniblock} "
                   f"or {rcvblocks.keyblock} is not normative")
            raise ValueError(err)
        if notReceived:
            random.shuffle(self.receive_tape)
        return notReceived
    


    def receive_miniblock(self, rcvblocks: NewBlocks):
        '''
        检查该miniblock解决的问题是否已经被接收过了

        pref: 从该miniblock所在的keyblock开始检查是否以接收过
        '''
        rcv_mb = rcvblocks.miniblock
        keyid = rcv_mb.get_keyid()
        if keyid == self.consensus.cur_keyid :
            if len(self.consensus.cur_keyblock.next) == 0:
                self.receive_tape.append(rcvblocks) 
                return True
            if rcv_mb not in self.local_chain.get_mbs_after_kb(
                self.consensus.cur_keyblock):
                self.receive_tape.append(rcvblocks) 
                return True
            return False
        # if keyprblm_id != self.consensus.KEY_PRBLM_ID
        local_kbs = self.local_chain.get_keyblocks_pref()
        for kb in local_kbs:
            if kb.keyfield.key_tx is None:
                continue
            if keyid != kb.get_keyid():
                continue
            if len(kb.next) == 0:
                self.receive_tape.append(rcvblocks) 
                return True
            if rcv_mb not in self.local_chain.get_mbs_after_kb(kb):
                self.receive_tape.append(rcvblocks)
                return True
            else:
                return False
        logger.warning("%s: not found the keyblock with keyid %s the miniblock "
                       "%s links to!" ,self.LOG_PREFIX, keyid, rcv_mb.name)
        return False

    def receive_keyblock(self, rcvblocks: NewBlocks):
        """
        不接收：
            已经接收到了完全相同的key problem transaction
            存在连接到相同keyblock的keyblock, 且接收到的块未包含新问题
        """
        rcv_kb = rcvblocks.keyblock
        local_kbs = self.local_chain.get_keyblocks_pref()
        for kb in local_kbs:
            # # 如果已经有了相同的问题，就不接收
            # if rcvkfield.orig_prblm_tx is not None:
            #     if rcvkfield.orig_prblm_tx == kbfield.orig_prblm_tx:
            #         return False
            # 如果上一个keyblock相同，但是接收到的区块没有包含下一个问题，就不接收
            if (kb.keyfield.pre_kb is not None and rcv_kb.keyfield.key_tx is None
                and rcv_kb.keyfield.pre_kb.name == kb.keyfield.pre_kb.name):
                return False
        if rcv_kb not in local_kbs:
            self.receive_tape.append(rcvblocks)
            if rcv_kb.keyfield.key_tx:
                self.txpool.reorg(rcv_kb.keyfield.key_tx)
            return True
        return False

    def mining(self, round):
        """mining interface

        Params:
            round(int): 当前的轮次数

        Returns:
            newblockss(list): 挖出的所有区块
            mine_success(bool): 挖矿成功标识
        """
        newblocks, mineSuccess = self.consensus.mining_consensus(
            self.local_chain, self.miner_id, self.isAdversary,
            self.input, self.q, round, self.txpool)
        if mineSuccess is False:
            return newblocks, mineSuccess
        # 成功挖出区块
        # keyblock
        if newblocks.iskeyblock:
            new_kb = newblocks.keyblock
            b2link = self.local_chain.search_forward_by_hash(new_kb.blockhead.prehash)
            self.local_chain.add_block_direct(new_kb, b2link)
            # 建立与与前一个keyblock的连接
            pre_kb = new_kb.keyfield.pre_kb
            pre_kb.keyfield.next_kbs.append(new_kb)
            logger.info("%s: mined and added keyblock %s with pre_keyblock %s, "
                        "next keyblocks of %s: %s", self.LOG_PREFIX, new_kb.name, 
                        new_kb.keyfield.pre_kb.name, new_kb.keyfield.pre_kb.name,
                        [b.name for b in pre_kb.keyfield.next_kbs])
            # 添加随keyblock一起挖出的miniblock
            if self.consensus.kb_strategy != 'pow':
                new_mb_with_kb = newblocks.mb_with_kb
                self.local_chain.add_block_direct(new_mb_with_kb, new_kb)
            return newblocks, mineSuccess
        # miniblock
        new_mb = newblocks.miniblock
        b2link = self.local_chain.search_forward_by_hash(new_mb.blockhead.prehash)
        self.local_chain.add_block_direct(new_mb, b2link)
        logger.info("%s: mined and added a miniblock %s , fathomd state %s",
                    self.LOG_PREFIX, new_mb.name, new_mb.get_fthmstat())
        # 更新forkview
        for forkview in self.forkviews:
            forkview.check_on_curfork_and_update_forks(new_mb, self.miner_id, self.round)
        # 如果该区块fathomed，更新问题树状态
        if new_mb.minifield.bfthmd_state:
            if new_mb.update_solve_tree_fthmd_state():
                k_fstat = self.consensus.cur_keyblock.get_fthmstat()
                logger.info("%s: updated solve tree with %s , key fathomd state %s", 
                            self.LOG_PREFIX, new_mb.name, k_fstat)
                self.consensus.reorg_open_blocks()
        # miniblock不满足安全性，仅保存在本地链上，不公布
        if not self.consensus.check_mb_safety(new_mb):
            logger.info("%s: %s not reach safety thre with theory attack rate %s", 
                        self.LOG_PREFIX, new_mb.name, new_mb.minifield.atk_rate)
            mineSuccess = False
            new_mb.ispublished = False
            return None, mineSuccess
        # `warning` 弃用dmin
        # if new_mb.get_prblm_depth() < self.consensus.dmin:
        #     mine_success = False
        #     return None, mine_success
        return newblocks, mineSuccess

    
    def record_new_fork_mb(self, root_block:Block, root_pname:tuple, 
                           cur_block:Block, cur_fork:Block =None):
        """ 依据root_block和root_pname记录一个fork，
        并指明目前所在的fork，保存到`fork_views`
        """
        if len(root_block.next) == 0:
            return
        # 如果该root已经被记录了，就更新一下
        isRecorded, forkview = self.root_already_recorded(root_pname)
        if isRecorded:
            forkview.add_new_forks(cur_block)
            logger.info("%s: adding a fork %s with root %s", 
                        self.LOG_PREFIX, cur_block.name, root_block.name)
            return
        # 如果未被记录，新建fork_view并保存
        forkview = ForkView(root_block, root_pname, cur_fork)
        # 添加fork，以fork的第一个block代表一个fork
        for b in root_block.next:
            if b.iskeyblock:
                return
            if b.minifield.pre_pname == root_pname:
                forkview.add_new_forks(b)
        if len(forkview.forks) <= 1:
            logger.warning("%s: set up forkview failed! root %s has no forks", 
                           self.LOG_PREFIX, root_block.name)
            return
        logger.info("%s: set up a forkview with root %s and forks %s, cur fork %s",
                    self.LOG_PREFIX, root_block.name, forkview.get_forks(), cur_fork.name)
        self.forkviews.append(forkview)
    
    def root_already_recorded(self, root_pname:tuple):
        """检查是否已经记录了这个root"""
        isRecorded = False
        if len(self.forkviews)==0:
            return isRecorded, None
        for fv in self.forkviews:
            if root_pname == fv.root_pname:
                isRecorded = True
                return  isRecorded, fv
        return isRecorded, None

    def fork_detect_mb(self, cur_block:Block):
        """
        检测是否有分叉，若有，则记录到`fork_views`；
        检测规则：即open_blocks中存在和该block链接到同一个子问题上的block

        TODO：修改检测逻辑，可能分叉部分已经不在open blocks中了
        """
        def detect_on_cur_fork(cur_block:Block):
            if len(self.forkviews) == 0:
                return None
            onCurFork = None
            for forkview in self.forkviews:
                onCurFork = forkview.check_on_curfork_and_update_forks(
                    cur_block, self.miner_id, self.round)
                logger.info("%s: %s onCurFork %s ", 
                            self.LOG_PREFIX, cur_block.name, onCurFork)
                if onCurFork is not None:
                    break
            return onCurFork
                
        def detect_new_fork(cur_block:Block):
            # 检测是否是新分叉
            if len(self.consensus.open_blocks)==0:
                return False
            pre_pname = cur_block.get_pre_pname()
            for ob in self.consensus.open_blocks:
                if ob.iskeyblock:
                    return False
                if pre_pname == ob.get_pre_pname():
                    logger.info("miner %s detected a fork %s with root %s", 
                                self.miner_id, cur_block.name, ob.pre.name)
                    # self.evaluation.record_fork_times(cur_block.get_block_time())
                    self.record_new_fork_mb(cur_block.pre, pre_pname, cur_block, ob)
                    return True
            return False
    
        if cur_block.iskeyblock:
            return False
        isNewFork = detect_new_fork(cur_block)
        if isNewFork:
            return True
        onCurFork = detect_on_cur_fork(cur_block)
        if onCurFork is not None:
            return not onCurFork
        return False
    
        
    def fork_switch_mb(self):
        """"""
        if len(self.forkviews) == 0:
            return
        del_fvs = []
        for fork_view in self.forkviews:
            switchFork, prefkbs, curfkbs = fork_view.fork_switch(
                self.miner_id, self.round)
            if not switchFork:
                continue
            del_fvs.append(fork_view)
            if self.consensus.pre_block in prefkbs:
                self.consensus.cancel_solving_prblm()
            logger.info("%s: switching fork... pre open blocks: %s, pre opt_prblms : %s",
                        self.LOG_PREFIX, [ob.name for ob in self.consensus.open_blocks],
                        [p.pname for p in self.consensus.opt_prblms])
            obs = [ob for ob in self.consensus.open_blocks if ob not in prefkbs]
            for fkb in curfkbs:
                self.consensus.check_again_mb_fthmd_state(fkb)
                if fkb.get_fthmstat():
                    if fkb.update_solve_tree_fthmd_state():
                        self.consensus.reorg_open_blocks()
            # TODO: 原分叉上找到了比新分叉上更优的问题的情况
            # 清除在原分叉上找到的最优问题
            del_optps = []
            for i, opt_p in enumerate(self.consensus.opt_prblms):
                for fkb in prefkbs:
                    if opt_p not in fkb:
                        continue
                    del_optps.append(opt_p)
            self.consensus.opt_prblms = [opt_p for opt_p in self.consensus.opt_prblms 
                                         if opt_p not in del_optps]
            lb = -sys.maxsize if len(self.consensus.opt_prblms) == 0 else self.consensus.opt_prblms[0].z_lp
            p:lpprblm.LpPrblm
            for fkb in curfkbs:
                for p in fkb:
                    if not (p.all_integer() and p.z_lp>=lb):
                        continue
                    if p.z_lp>lb:
                        self.consensus.opt_prblms.clear()
                    self.consensus.opt_prblms.append(p)

            obs.extend([ob for ob in curfkbs if not ob.get_fthmstat()])
            self.consensus.open_blocks = obs
            logger.info("%s: switched fork with cur open blocks: %s, cur opt_prblms: %s ",
                        self.LOG_PREFIX, [ob.name for ob in self.consensus.open_blocks],
                        [p.pname for p in self.consensus.opt_prblms])
        # 切换fork后直接删除该fork view
        self.forkviews = [fv for fv in self.forkviews if fv not in del_fvs]


    def fork_choice(self, round):
        # print(self.consensus.KEY_PRBLM_ID)
        new_update = False
        if len(self.receive_tape) == 0:
            return self.local_chain, new_update
        
        for rcvblocks in self.receive_tape:
            if not rcvblocks.iskeyblock:      
                self.fork_choice_miniblock(round, rcvblocks)
            elif rcvblocks.iskeyblock:
                self.fork_choice_keyblock(round, rcvblocks)
            else:
                raise ValueError(f"Miner{self.miner_id}: The rceived block "
                    f"{rcvblocks.miniblock} or {rcvblocks.keyblock} is not normative")
        return self.local_chain, new_update
        

    def fork_choice_miniblock(self, round, rcvblock:NewBlocks):
        """
        rcvblock与正在求解的miniblock连接同一个子问题，放弃当前求解
        open_blocks中有区块与rcvblock连接到相同的问题，就不把rcvblock放入open_blocks中

        如果求解树的状态有更新，将连接到fthmd_state的区块从待求解问题集中删去，
        同时当前求解若连接到fthmd_state区块，也终止求解

        param:  rcvblocks: 接收到的区块列表，其中应当只有一个miniblock
        return: new_update(bool): 本地链是否有更新
        """
        new_update = False
        if rcvblock.iskeyblock:
            raise ValueError("Received block is not a miniblock")
        rcv_mb = rcvblock.miniblock
        logger.info("round %s miner %s: rcv_mb %s fthmd state %s",
                    round, self.miner_id, rcv_mb.name, rcv_mb.get_fthmstat())
        # 验证合法性
        if not self.consensus.valid_block(self.local_chain, rcv_mb):
            return new_update
        # 添加到链上
        copy_mb, add_success = self.add_miniblock_to_local_chain(rcv_mb)
        if add_success is False:
            return new_update
        new_update = True
        self.consensus.check_again_mb_fthmd_state(copy_mb)
        key_id = copy_mb.get_keyid()
        # 如果和当前求解的keyblock不一致，不进行更多操作
        if key_id != self.consensus.cur_keyid:
            return new_update
        # 如果当前没有正在求解的keyblock，不进行更多操作
        if self.consensus.cur_keyblock is None:
            return new_update
        # 如果当前keyblock已求解完，不进行更多操作
        key_prblm = self.consensus.cur_keyblock.get_keyprblm()
        if key_prblm.fthmd_state is True:
            return new_update
        # 如果和正在求解的区块连接的问题相同，放弃当前求解
        if self.consensus.pre_prblm is not None:
            if self.consensus.pre_prblm.pname == copy_mb.minifield.pre_pname:
                unpub_num, unpubs = self.consensus.get_unpub_num(False, self.evaluation.recordSols)
                # unpub_time = rcv_mb.get_block_time() if unpub_num > 0 else None
                # logger.info("round %s, miner%s: unpub pair num %s, block time %s",
                #             round, self.miner_id, unpub_num, unpub_time)
                self.evaluation.record_unpub_pair(
                    self.consensus.cur_keyblock.name, unpub_num, 
                    unpub_subs = unpubs, miner_id= self.miner_id)
                self.consensus.cancel_solving_prblm()
        # 检测分叉
        if self.fork_detect_mb(copy_mb) is False:
            self.consensus.open_blocks.append(copy_mb) 
        if self.fork_switch_mb():
            return 
        logger.info("round %s miner %s: copy_mb %s fthmd state %s",
                    round, self.miner_id, copy_mb.name, copy_mb.get_fthmstat())
        # 更新状态树，重组
        if copy_mb.get_fthmstat() and copy_mb.update_solve_tree_fthmd_state():
            self.consensus.reorg_open_blocks()
            logger.info("%s: updating solve tree miniblock %s fathomed_state: %s, "
                        "keyblock fathomed: %s", self.LOG_PREFIX, copy_mb.name, 
                        copy_mb.get_fthmstat(), key_prblm.fthmd_state)
        
        return new_update

    def add_miniblock_to_local_chain(self, rcv_mb:Block):
        """
        将miniblcok复制后添加到本地链上
        return: 复制过后的区块
        """
        copy_mb = copy.deepcopy(rcv_mb)
        add_success = False
        b2link = self.local_chain.search_forward_by_hash(rcv_mb.blockhead.prehash)
        if b2link is None:
            logger.warning("%s: add block %s failed, not found %s", 
                           self.LOG_PREFIX, rcv_mb.name, rcv_mb.get_prehash())
            return copy_mb, add_success
        add_success = True
        self.local_chain.add_block_direct(copy_mb, b2link)
        # 与该区块branch的子问题建立连接
        for prblm in b2link:
            if prblm.pname == rcv_mb.minifield.pre_pname:
                copy_mb.minifield.pre_p = prblm
        return copy_mb, add_success

         
    def fork_choice_keyblock(self, round, rcvblock:NewBlocks):
        new_update = False
        if not rcvblock.iskeyblock:
            raise ValueError("Received block is not a keyblock")
        # 先处理不满足安全性的miniblock和keyblock
        rcvkb = rcvblock.keyblock
        rcvmbs_unsafe = rcvblock.mbs_unsafe
        # 验证合法性
        for mb_unsafe in rcvmbs_unsafe:
            if not self.consensus.valid_block(self.local_chain, mb_unsafe):
                return new_update
        if not self.consensus.valid_block(self.local_chain, rcvkb):
                return new_update
        # 先添加不满足安全性的miniblock到链上
        for mb_unsafe in rcvmbs_unsafe:
            copyblk, add_success = self.add_miniblock_to_local_chain(mb_unsafe)
            if add_success is False:
                return new_update
            if copyblk.minifield.bfthmd_state:
                copyblk.update_solve_tree_fthmd_state()
        # 再添加keyblock在链上
        copykb = copy.deepcopy(rcvkb)
        b2link = self.local_chain.search_forward_by_hash(rcvkb.blockhead.prehash)
        if b2link is None:
            logger.warning("%s: add key-block %s failed, not found %s",
                           rcvkb.name, rcvkb.blockhead.prehash)
            return new_update
        # 建立keyblock间的连接
        pre_kb = rcvkb.keyfield.pre_kb
        if pre_kb is not None:
            local_kbs = self.local_chain.get_keyblocks_pref()
            for kb in local_kbs:
                if pre_kb.get_hash() != kb.get_hash():
                    continue
                copykb.keyfield.pre_kb = kb
                kb.keyfield.next_kbs.append(copykb)
        logger.info("%s: add keyblock %s success, pre kb %s, next kbs of %s: %s",
                    self.LOG_PREFIX, copykb.name, copykb.keyfield.pre_kb.name, 
                    copykb.keyfield.pre_kb.name, [b.name for b in kb.keyfield.next_kbs])
        self.local_chain.add_block_direct(copykb, b2link)
        new_update = True
        # 如果策略为'pow'
        if self.consensus.kb_strategy == 'pow':
            # 如果该keyblock的高度高于正在求解的keyblock，则切换到该问题
            if copykb.get_keyheight() > self.consensus.cur_keyblock.get_keyheight():
                unpub_num, unpubs = self.consensus.get_unpub_num(True, self.evaluation.recordSols)
                self.evaluation.record_unpub_pair(
                    self.consensus.cur_keyblock.name, unpub_num, 
                    unpub_subs = unpubs, miner_id= self.miner_id)
                self.consensus.switch_key(copykb,round=round)
            return new_update

        # 如果策略为withmini 或 'pow+withmini'
        rcvmb_with_kb = rcvblock.mb_with_kb
        if not self.consensus.valid_block(self.local_chain, rcvmb_with_kb):
            return new_update
        copy_mbwithkb, add_success = self.add_miniblock_to_local_chain(rcvmb_with_kb)
        if add_success is False:
            return new_update
        if self.consensus.withmini_keycache is not None:
            if (copykb.get_keyheight() >= 
                self.consensus.withmini_keycache.get_keyheight()):
                unpub_num, unpubs = self.consensus.get_unpub_num(True, self.evaluation.recordSols)
                self.evaluation.record_unpub_pair(
                    self.consensus.cur_keyblock.name, unpub_num, 
                    unpub_subs = unpubs, miner_id= self.miner_id)
                self.consensus.switch_key(copykb,round=round)
                self.consensus.withmini_keycache = None
                self.consensus.open_blocks.append(copy_mbwithkb)
        else:
            if (copykb.get_keyheight() > 
                self.consensus.cur_keyblock.get_keyheight()):
                unpub_num, unpubs = self.consensus.get_unpub_num(True, self.evaluation.recordSols)
                self.evaluation.record_unpub_pair(
                    self.consensus.cur_keyblock.name, unpub_num,
                    unpub_subs = unpubs, miner_id= self.miner_id)
                self.consensus.switch_key(copykb,round=round)
                self.consensus.open_blocks.append(copy_mbwithkb)
        
        if copy_mbwithkb.minifield.bfthmd_state:
                copy_mbwithkb.update_solve_tree_fthmd_state()
        
        return new_update


    def backbone_protocol(self, round):
        self.round = round
        self.LOG_PREFIX = f"round {self.round} miner {self.miner_id}"
        self.consensus.update_round(round)
        chain_update, update_index = self.fork_choice(round)
        newblocks, mine_success = self.mining(round)
        if mine_success:
            newblocks = newblocks
        if update_index or mine_success:  # Cnew != C
            return newblocks
        else:
            return None  # 如果没有更新 返回None告诉environment回合结束

if __name__ == "__main__":
    class A(object):
        def __init__(self) -> None:
            self.aa = defaultdict(list[B])
    class B(object):
        def __init__(self, bb) -> None:
            self.bb = bb
        def __repr__(self) -> str:
            return str(self.bb)
    lowest_ops = [B(2),B(1),B(3),B(5),B(4)]
    open_prblm = random.choice(lowest_ops)
    pop_idx = lowest_ops.index(open_prblm)
    s = str(None)
    print(s, type(s))
    # a = A()
    # b = B(1)
    # a.aa['a'].append(B(1))
    # a.aa['a'].append(B(2))
    # a.aa['b'].append(2)
    # for fork, forkblocks in a.aa.items():
    #     for fb in forkblocks:
    #         print(fb.bb)
    #     print(fork,forkblocks)
    a = []
     # print(a.aa, a.bb)
    # print(b.aa)