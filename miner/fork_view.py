import logging
from collections import defaultdict

from data import Block

logger = logging.getLogger(__name__)

FORK_SWITCH_THRE = 1

class ForkView(object):
    # TODO expire
    def __init__(self,  root_block:Block, root_pname:tuple, cur_fork:Block) -> None:
        """
        cur_fork: 使用fork上的第一个block代表该fork
        forks：记录各个分叉，结构
        {
            b1_on_fork1.name: [b1_on_fork1, b2_on_fork1...],
            b1_on_fork2.name: [b1_on_fork2, b2_on_fork2...],
        }
        """
        self.root_block = root_block
        self.root_pname = root_pname
        self.cur_fork = cur_fork
        self.forks = defaultdict(list[Block])
        self.expired = False

    def get_forks(self):
        return list(self.forks.keys())
    
    def add_new_forks(self, *forks:Block):
        for fork in forks:
            self.forks[fork.name] = [fork]

    def check_on_curfork_and_update_forks(self, block:Block, miner_id, round):
        onCurFork = True
        log_prefix = f"round {round} miner {miner_id}"
        logger.info("%s: checking onCurFork: %s, pre %s, cur fork: %s", 
                    log_prefix, block.name, block.pre.name, self.cur_fork.name)
        for forkname, forkblocks in self.forks.items():
            if block in forkblocks:
                logger.info("%s: %s is already on fork %s", 
                            log_prefix, block.name, forkname)
                onCurFork = True if forkname == self.cur_fork.name else False
                return onCurFork
            if block.pre in forkblocks:
                self.forks[forkname].append(block)
                logger.info("%s: detect a new block %s on fork %s with %s", log_prefix, 
                            block.name, forkname, [b.name for b in self.forks[forkname]])
                onCurFork = True if forkname == self.cur_fork.name else False
                return onCurFork
        return None
    
    def get_fork_block(self, fork_name:str):
        for fb in self.forks[fork_name]:
            if fb.name == fork_name:
                return fb
        return None

    
    def get_entire_fork(self, fork_block:Block):
        q:list[Block] = [fork_block]
        entire_fork:list[Block] = []
        while q:
            b = q.pop(0)
            if not b.iskeyblock:
                entire_fork.append(b)
            if len(b.next) == 0:
                continue
            q.extend([b for b in b.next if not b.iskeyblock])
        return entire_fork
    

    
    def fork_switch(self, miner_id, round):
        log_prefix = f"round {round} miner {miner_id}"
        switchFork = False
        hvest = self.cur_fork.get_heavy()
        hvest_fork = self.cur_fork
        pre_forkbs:list[Block] = []
        hvest_forkbs:list[Block] = []
        # 找到heavy最大的分叉
        for fb in self.forks.keys():
            hv = self.get_fork_block(fb).get_heavy() 
            if hv - hvest < FORK_SWITCH_THRE:
                continue
            hvest = hv
            hvest_fork = self.get_fork_block(fb)
        if hvest_fork.get_hash() == self.cur_fork.get_hash():
            return switchFork, pre_forkbs, hvest_forkbs
        
        switchFork = True
        logger.info("%s: switch to a new fork %s with heavy %s, pre fork %s "
                    "with heavy %s", log_prefix, hvest_fork.name, hvest, 
                    self.cur_fork.name, self.cur_fork.get_heavy())
        pre_forkbs = self.get_entire_fork(self.cur_fork)
        self.cur_fork = hvest_fork
        hvest_forkbs = self.get_entire_fork(self.cur_fork)
        return switchFork, pre_forkbs, hvest_forkbs
