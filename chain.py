import copy
import logging
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass

import graphviz
import matplotlib.pyplot as plt

import lpprblm
from background import Background
from functions import hashG, hashH
from lpprblm import LpPrblm
from txpool import Transaction

__all__ = ['BlockHead', 'Block', 'KeyField', 'MiniField', 'Chain']
logger = logging.getLogger(__name__)

@dataclass
class NewBlocks:
    """
    共识产生的新区块的结构
    """
    iskeyblock:bool
    miniblock:'Block'
    keyblock:'Block'
    mbs_unsafe:list['Block']
    mb_with_kb:'Block'

class BlockHead(object):
    def __init__(self, prehash = None, blockhash = None, 
                 height = None, miner_id = None, timestamp = None):
        '''
        :param prehash: 上一个区块的哈希, 在branchbound中姑且用上个区块的块名
        :param blockhash: 本区块的哈希, 在branchbound中姑且用本区块的块名
        :param height: 区块高度
        :param miner_id: 产生该区块的矿工
        :param timestamp: 时间戳信息, 产生该区块的轮次
        '''
        self.prehash = prehash  # 前一个区块的hash
        self.height = height  # 高度
        self.heavy = 1 # 连接到该区块的子区块数量
        self.blockhash = blockhash  # 区块哈希
        self.miner_id = miner_id  # 矿工
        self.timestamp = timestamp # 时间戳信息, 产生该区块的轮次
    
    def __deepcopy__(self, memo):
        """copy时设置heavy为0
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))
        setattr(result, "heavy", 1)
        return result
        
        
    def __repr__(self) -> str:
        bhlist = []
        omit_keys = {'nonce', 'blockheadextra', 'target'}
        for k, v in self.__dict__.items():
            if k not in omit_keys:
                bhlist.append(k + ': ' + str(v))
        return '\n'.join(bhlist)


class KeyField(object):
    """
    保存keyblock的信息
    """
    def __init__(self, 
            key_hash:str = None, 
            pow_nonce:int = None, 
            key_height:int = None,
            pre_keyblock:"Block" = None, 
            pre_pname:tuple = None, 
            pre_key_feasible:bool = None, 
            opt_prblms:list[LpPrblm] = None, 
            fthmd_prblms:list[LpPrblm] = None,
            key_tx:Transaction = None,
            accept_mbs:list[str] = None
        ):
        # Key PoW
        self.key_hash = key_hash
        self.pow_nonce = pow_nonce
        self.key_height = key_height
        # some links
        self.pre_kb = pre_keyblock
        self.pre_pname = pre_pname
        self.next_kbs:list[Block] = []
        # Related to the previous keyblock
        self.pre_key_feasible = pre_key_feasible
        self.opt_prblms = opt_prblms
        self.fthmd_prblms = fthmd_prblms
        self.accept_mbs = accept_mbs
        # New key problem
        self.key_tx = key_tx

        # thres
        self.thres = None

    def init_thres(self, thres:list):
        self.thres = thres

    def __repr__(self) -> str:
        pre_kb_name = self.pre_kb.name \
                    if self.pre_kb is not None else None
        fthmd_prblms = [p.pname for p in self.fthmd_prblms] \
                    if len(self.fthmd_prblms)!=0 else []
        opt_prblm_names = [p.pname for p in self.opt_prblms] \
                    if len(self.opt_prblms) > 0 else []
        next_keyblocks = [b.name for b in self.next_kbs] \
                    if len(self.next_kbs) > 0 else []        
        key_str = (f"key_hash: {self.key_hash}\n"+
                   f"key_height:{self.key_height}\n"+
                   f"pow_nonce: {self.pow_nonce}\n"+
                   f"origin_prblm: {self.key_tx}\n" + 
                   f"pre_keyblock: {pre_kb_name}\n"+
                   f"pre_key_feasible: {self.pre_key_feasible}\n"+
                   f"next_keyblocks:{next_keyblocks}\n"
                   f"fthmd_prblms: {fthmd_prblms}\n"+
                   f"opt_prblm: {opt_prblm_names}\n")
        return key_str

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if (cls.__name__ == 'KeyField' and k != 'pre_keyblock' 
                                        and k != 'next_keyblocks'):
                setattr(result, k, copy.deepcopy(v, memo))
            if cls.__name__ == 'KeyField' and k == 'pre_keyblock':
                setattr(result, k, None)
            if cls.__name__ == 'KeyField' and k == 'next_keyblocks':
                setattr(result, k, [])
            if cls.__name__ != 'KeyField':
                setattr(result, k, copy.deepcopy(v, memo))
        return result
        
class MiniField(object):
    """
    保存miniblock的信息
    """
    def __init__(self):
        self.pre_pname:tuple = None
        self.pre_p:LpPrblm = None
        self.subprblm_pairs:list[tuple[LpPrblm, LpPrblm]] = []
        # pair node classification
        self.root_pair:tuple[LpPrblm, LpPrblm] = None
        self.leaf_prblms:list[LpPrblm] = []
        self.deepest_prblms:list[LpPrblm] = []
        # extra info just benefit for programming
        # 如果所有叶子节点都fathomed，则该miniblock也fathomed
        self.atk_rate = 0
        # extra info just benefit for programming(local state)
        self.bfthmd_state = False
    
    def __iter__(self)->tuple[LpPrblm,LpPrblm]:
        if len(self.subprblm_pairs) == 0:
            raise ValueError("No subproblem pair in the miniblock")
        for subprblm_pair in self.subprblm_pairs:
            if len(subprblm_pair) != 2:
                raise ValueError("The structure of the miniblock error!")
            else:
                yield subprblm_pair

    def __deepcopy__(self,memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if cls.__name__ == 'MiniField' and k != 'pre_p' and k!= 'bfthmd_state':
                setattr(result, k, copy.deepcopy(v, memo))
            if cls.__name__ == 'MiniField' and k == 'pre_p':
                setattr(result, k, None)
            if cls.__name__ == 'MiniField' and k == 'bfthmd_state':
                setattr(result, k, False)
            if cls.__name__ != 'MiniField':
                setattr(result, k, copy.deepcopy(v, memo))
        return result
        
        
    def add_subprblm_pair(self, subprblm1:LpPrblm, subprblm2:LpPrblm):
        if subprblm1.pre_pname != subprblm2.pre_pname:
            raise ValueError(f"Add subprblm pair error! {subprblm1.pre_pname}"
                                 f"!= {subprblm2.pre_pname}")
        self.subprblm_pairs.append((subprblm1, subprblm2))

    def classify_nodes(self):
        """
        将miniblock中的所有`子问题`点分为根节点、叶子节点和最深节点
        """
        # if len(self.subprblm_pairs) == 0:
        #     raise ValueError("miniblock have no subproblems")

        self.root_pair = None
        self.leaf_prblms.clear()
        self.deepest_prblms.clear()

        not_leaf = []
        min_depth = self.subprblm_pairs[0][0].pheight
        max_depth = self.subprblm_pairs[0][0].pheight
        # 根节点和最深节点
        for (p1,p2) in self.subprblm_pairs:
            not_leaf.extend([p1.pre_pname,p2.pre_pname])
            if p1.pheight <= min_depth:
                min_depth = p1.pheight
                self.root_pair = (p1,p2)
            if p1.pheight == max_depth:
                self.deepest_prblms.extend([p1,p2])
            if p1.pheight > max_depth:
                max_depth = p1.pheight
                self.deepest_prblms.clear()
                self.deepest_prblms.extend([p1,p2])
        # 叶子节点
        for (p1,p2) in self.subprblm_pairs:
            if p1.pname not in not_leaf:
                self.leaf_prblms.append(p1)
            if p2.pname not in not_leaf:
                self.leaf_prblms.append(p2)
        return self.root_pair, self.leaf_prblms, self.deepest_prblms
    
    def update_fathomed_state(self):
        """
        更新miniblock的区块状态
        """
        if len(self.leaf_prblms) == 0:
            self.classify_nodes()
        for prblm in self.leaf_prblms:
            if not prblm.fthmd_state:
                self.bfthmd_state = False
                return self.bfthmd_state
        self.bfthmd_state = True
        return self.bfthmd_state
        
    
    def __repr__(self) -> str:
        omit_keys = ["pre_p"]
        def _dict_formatter(d, mplus=1):
            if isinstance(d, dict) and len(d) != 0:
                m = max(map(len, list(d.keys()))) + mplus  # width to print keys
                s = '\n'.join([k.rjust(m) + ': ' +  # right justified, width m
                            _indenter(_dict_formatter(v, 0), m+2)
                            for k, v in d.items()])  # +2 for ': '
            elif isinstance(d, dict) and len(d) == 0:
                    s = '{}'
            else:
                s = str(d)
            return s
        def _indenter(s, n=0):
            split = s.split("\n")
            indent = " "*n
            return ("\n" + indent).join(split)
        
        bdict = copy.deepcopy({k:v for k,v in self.__dict__.items() if k not in ['pre_p']})
        # for omk in omit_keys:
        #     del bdict[omk]
        bdict["root_pair"] = (bdict["root_pair"][0].pname, bdict["root_pair"][1].pname)
        bdict["leaf_prblms"] = [p.pname for p in bdict["leaf_prblms"]]
        bdict["deepest_prblms"] = [p.pname for p in bdict["deepest_prblms"]]
        return '\n'+ _dict_formatter(bdict)

class Block(object):
    def __init__(self, name:str = None, iskeyblock:bool = None, 
                blockhead: BlockHead = None, content=None, 
                isadversary=False, isgenesis=False, blocksize_MB=2,
                ispublished=True):
        self.name = name
        self.blockhead = blockhead
        self.isAdversary = isadversary
        self.content = content
        self.isGenesis = isgenesis
        self.blocksize_MB = blocksize_MB
        # pointers
        self.next:list[Block] = []  # 子块列表
        self.pre:Block = None  # 母块
        # branchbound relative
        self.iskeyblock = iskeyblock
        self.keyfield:KeyField = None
        self.minifield:MiniField = None
        # extra info just benefit for programming(local state)
        self.ispublished = ispublished
        self.isFork = True
    
    def __iter__(self)->LpPrblm:
        if self.iskeyblock:
            if self.keyfield.key_tx is not None:
                yield self.keyfield.key_tx.data
            else:
                raise ValueError("No subproblem pair in the keyblock")
        else:
            if len(self.minifield.subprblm_pairs) == 0:
                raise ValueError("No subproblem pair in the miniblock")
            else:
                for (p1,p2) in self.minifield.subprblm_pairs:
                    yield p1
                    yield p2

    def __contains__(self, prblm: LpPrblm):
        if self.iskeyblock:
            if prblm == self.get_keyprblm():
                return True
            return False
        else:
            if len(self.minifield.subprblm_pairs) == 0:
                return False
            for (p1,p2) in self.minifield:
                if prblm == p1 or prblm == p2:
                    return True
        return False


    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if cls.__name__ == 'Block' and k != 'next' and k != 'pre':
                setattr(result, k, copy.deepcopy(v, memo))
            if cls.__name__ == 'Block' and k == 'next':
                setattr(result, k, [])
            if cls.__name__ == 'Block' and k == 'pre':
                setattr(result, k, None)
            if cls.__name__ != 'Block':
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __repr__(self) -> str:
        omit_keys = {'isAdversary',  'blocksize_MB'}
        def _dict_formatter(d, mplus=1):
            if isinstance(d, dict) and len(d) != 0:
                m = max(map(len, list(d.keys()))) + mplus  # width to print keys
                s = '\n'.join([k.rjust(m) + ': ' +  # right justified, width m
                            _indenter(_dict_formatter(v, 0), m+2)
                            for k, v in d.items()])  # +2 for ': '
            elif isinstance(d, dict) and len(d) == 0:
                    s = '{}'
            else:
                s = str(d)
            return s
        def _indenter(s, n=0):
            split = s.split("\n")
            indent = " "*n
            return ("\n" + indent).join(split)
        
        bdict = copy.deepcopy({k:v for k,v in self.__dict__.items() if k not in ['next','pre']})
        bdict.update({'next': [b.name for b in self.next if self.next], 
                      'pre': self.pre.name if self.pre is not None else None})
        if self.iskeyblock:
            del bdict['minifield']
        else:
            del bdict['keyfield']
        for omk in omit_keys:
            del bdict[omk]
        return '\n'+ _dict_formatter(bdict)


    def get_hash(self):
        return self.blockhead.blockhash
    
    def get_prehash(self):
        return self.blockhead.prehash

    def get_height(self):
        """获取区块的高度"""
        return self.blockhead.height
    
    def get_heavy(self):
        """获取区块的heavy"""
        return self.blockhead.heavy
    
    def inc_heavy(self):
        """增加区块的heavy"""
        self.blockhead.heavy += 1
        # logger.info(f"{self.name}: updated heavy: {self.get_heavy()}")

    def get_miner_id(self):
        """获取miner_id"""
        return self.blockhead.miner_id
    
    def get_block_time_w_pre(self):
        """"""
        if self.pre is None:
            return 0
        return self.get_timestamp()-self.pre.get_timestamp()
    
    def get_timestamp(self):
        return self.blockhead.timestamp
    
    def get_kb_time_w_next(self):
        """获取与下一个keyblock的时间差"""
        if not self.iskeyblock:
            raise ValueError(f"get kb time error: {self.name} is a miniblock")
        next_kbs:list[Block] = sorted(self.keyfield.next_kbs, 
                                      key=lambda x: x.get_timestamp())
        kb_time = next_kbs[0].get_timestamp() - self.get_timestamp()
        return kb_time

    def get_subpairs_name_in_mb(self)->list[tuple]:
        if self.iskeyblock:
            logger.warning("Getting subpairs: %s is a key-block", self.name)
            return []
        if len(self.minifield.subprblm_pairs)==0:
            logger.warning("%s donot have subpairs", self.name)
            return []
        return [(p1.pname, p2.pname) for (p1,p2) in self.minifield]
    
    def get_pre_pname(self):
        if self.iskeyblock:
            return self.keyfield.pre_pname
        return self.minifield.pre_pname
    
    def get_fthmstat(self):
        """获取区块的fathomed state"""
        if self.iskeyblock:
            if self.keyfield.key_tx is None:
                logger.warning("%s get fathomed state failed", self.name)
                return None
            return self.keyfield.key_tx.data.fthmd_state
        return self.minifield.bfthmd_state

    def get_keyheight(self):
        """获取由keyblock组成的链中，该keyblock的高度"""
        if not self.iskeyblock:
            raise TypeError("get keyheight: %s is not a keyblock", self.name)
        return self.keyfield.key_height

    def get_keyprblm(self):
        """get the origin key problem in the keyblock

        Returns:
            orgin_prblm(LpPrblm): If the keyblock contains a orgin_prblm 
            else None
        """
        if self.iskeyblock is False:
            raise TypeError(f'{self.name} is not a keyblock')
        if self.keyfield.key_tx is None:
            return None
        return self.keyfield.key_tx.data

    def get_keypname(self):
        """获取keyblock中key_problem的name"""
        kp = self.get_keyprblm()
        kpname = kp.pname if kp is not None else None
        return kpname
    
    def get_keyid(self):
        """获取该block所在key的id, 如果是keyblock且没有包含key problem，返回None"""
        if self.iskeyblock:
            keyprblm = self.get_keyprblm()
            if keyprblm is None:
                logger.warning("get keyid of %s failed: key problem is None", self.name)
                return None
            return keyprblm.pname[0][0]
        return self.minifield.root_pair[0].pname[0][0]

    
    
    def get_soltree_depth(self):
        """
        获取block中问题的深度, 若keyblock返回1, Miniblock返回高度差
        """
        # keyblock
        if self.iskeyblock:
            logger.warning(f"{self.name} is a keyblock, the layer num returns 1")
            return 1
        # miniblock
        if len(self.minifield.subprblm_pairs)==0:
            raise ValueError("Do not contain any sub-problems")
        if len(self.minifield.deepest_prblms) == 0 or self.minifield.root_pair is None:
            raise ValueError("Do not contain any sub-problems")
        return (self.minifield.deepest_prblms[0].pheight 
                    - self.minifield.root_pair[0].pheight + 1)

    
    def get_deepest_subprblms(self):
        """
        获取最深的子问题集合
        """
        # keyblock
        if self.iskeyblock:
            if self.keyfield.key_tx is None:
                raise ValueError(f"keyblock {self.name} have no origin problem")
            return [self.keyfield.key_tx.data]
        # miniblock
        if len(self.minifield.subprblm_pairs) == 0:
            raise ValueError(f"miniblock {self.name} have no subproblems")
        elif len(self.minifield.deepest_prblms) == 0:
            self.minifield.classify_nodes()
        return self.minifield.deepest_prblms

    def get_unfthmd_leafps(self):
        """
        获取未被探明的子问题集合
        """
        # keyblock
        if self.iskeyblock:
            if self.keyfield.key_tx is None:
                raise ValueError(f"keyblock {self.name} have no origin problem")
            if self.keyfield.key_tx.data.fthmd_state:
                wmsg = (f"get_unfthmdstat_subpblms: keyblock {self.name} "
                        "is already fathomed")
                logger.warning(wmsg)
                return []
            return [self.keyfield.key_tx.data]
        # miniblock
        if len(self.minifield.subprblm_pairs) == 0:
            raise ValueError(f"miniblock {self.name} have no subproblems")
        elif len(self.minifield.leaf_prblms) == 0:
            self.minifield.classify_nodes()
        return [p for p in self.minifield.leaf_prblms if not p.fthmd_state]
                
                        
    def set_keyfield(self, 
            key_hash:str = None, 
            pow_nonce:int = None,
            key_height:int = None, 
            pre_keyblock:'Block' = None, 
            pre_pname:tuple = None, 
            pre_key_feasible:bool = None, 
            opt_prblms:list[LpPrblm] = None,
            fthmd_prblms:list[LpPrblm] = None, 
            keyprblm_tx:Transaction = None,
            accept_mbs:list[str] = None
        ):
        '''set the keyfield, the last_keyblock is a Block type'''
        if self.iskeyblock is False:
            raise TypeError(f'{self.name} is not a keyblock')
        if opt_prblms is None:
            opt_prblms = []
        if fthmd_prblms is None:
            fthmd_prblms = []
        self.keyfield =  KeyField(key_hash, pow_nonce, key_height, 
                pre_keyblock, pre_pname, pre_key_feasible, 
                opt_prblms, fthmd_prblms, keyprblm_tx, accept_mbs)
    
            
            
    def set_minifield(self, pre_prblm:LpPrblm, 
                    subprblm_pairs:list[tuple[LpPrblm,LpPrblm]]):
        if self.iskeyblock is True:
            raise TypeError(f'{self.name} is not a miniblock')
        if len(subprblm_pairs) == 0:
            raise TypeError("The list of subprblm pairs is empty")
        self.minifield = MiniField()
        for (p1, p2) in subprblm_pairs:
            self.minifield.add_subprblm_pair(p1,p2)
        self.minifield.classify_nodes()
        self.minifield.update_fathomed_state()
        self.minifield.pre_pname = pre_prblm.pname
        self.minifield.pre_p = pre_prblm

    def update_solve_tree_fthmd_state(self):
        """更新问题树的状态"""
        if self.iskeyblock:
            raise ValueError("update_solve_tree_fthmd_state: "
                            f"{self.name} not a miniblock")
        updateSuccess = False
        def update_state(block: Block):
            # 更新区块状态
            nonlocal updateSuccess
            if block.iskeyblock:
                return
            block.minifield.update_fathomed_state()
            if (block.minifield.bfthmd_state is True
                    and block.minifield.pre_p.fthmd_state is False):
                block.minifield.pre_p.fthmd_state = True
                updateSuccess = True
                update_state(block.pre)

        update_state(self)
        return updateSuccess
    
    def check_link_fthmd_prblm(self):
        """
        检查一个miniblock是否连接到了一个fathomed state的问题，
        如果连接到了返回True，否则返回False

        param: miniblock (Block): 代检查的miniblock
        returns: bool: True 连接到了一个fathomed state的问题
        """
        link_fthmd_flag = False

        def check(block: Block):
            nonlocal link_fthmd_flag
            if not block.iskeyblock:
                if not block.minifield.pre_p.fthmd_state:
                    check(block.pre)
                else:
                    link_fthmd_flag = True
            else:
                key_prblm = block.keyfield.key_tx.data
                if key_prblm.fthmd_state:
                    link_fthmd_flag = True

        check(self)
        return link_fthmd_flag
        
        
    def calculate_blockhash(self):
        '''
        计算区块的hash
        return:
            hash type:str
        '''
        content = self.content
        prehash = self.blockhead.prehash
        # nonce = self.blockhead.nonce
        # target = self.blockhead.target
        minerid = self.blockhead.miner_id
        hash = hashH([minerid, hashG([prehash, content])])  # 计算哈希
        return hash


class Chain(object):
    def __init__(self, background:Background, miner_id = None):
        self.background = background
        if miner_id is not None:
            self.miner_id = miner_id
        else: 
            self.miner_id = -1
        self.head = None
        self.lastblock = self.head  # 指向最新区块，代表矿工认定的主链
        self.lastkeyblock = self.head
        self.keyblocks:list[Block] = []
        self.main_chain = defaultdict(self.default_list)
        

    @ staticmethod
    def default_list():
        block_list:list[Block] = []
        return block_list

    def __contains__(self, block: Block):
        if not self.head:
            return False
        q = [self.head]
        while q:
            blocktmp = q.pop(0)
            if block.get_hash() == blocktmp.get_hash():
                return True
            for i in blocktmp.next:
                q.append(i)
        return False

    def __iter__(self):
        if not self.head:
            return
        q = [self.head]
        while q:
            blocktmp = q.pop(0)
            yield blocktmp
            for i in blocktmp.next:
                q.append(i)

    def __deepcopy__(self, memo):
        if not self.head:
            return None
        copy_chain = Chain(self.background, self.miner_id)
        copy_chain.head = copy.deepcopy(self.head)
        memo[id(copy_chain.head)] = copy_chain.head
        q = [copy_chain.head]
        q_o = [self.head]
        copy_chain.lastblock = copy_chain.head
        while q_o:
            for block in q_o[0].next:
                copy_block = copy.deepcopy(block, memo)
                copy_block.pre = q[0]
                q[0].next.append(copy_block)
                q.append(copy_block)
                q_o.append(block)
                memo[id(copy_block)] = copy_block
                if block.name == self.lastblock.name:
                    copy_chain.lastblock = copy_block
            q.pop(0)
            q_o.pop(0)
        return copy_chain
    
    
    def create_genesis_block(self, genesis_prblm :LpPrblm):
        prehash = 0
        height = 0
        miner_id = -1  # 创世区块不由任何一个矿工创建
        input = 0
        key_hash = ''
        pow_nounce = 0
        currenthash = 'B0'
        # lpprblm.solve_ilp_by_pulp(genesis_prblm)
        genesis_prblm.timestamp = 0
        genesis_prblm_tx = Transaction('env', 0, genesis_prblm)
        self.head = Block('B0', True, BlockHead(prehash, currenthash, height, 
                          miner_id, 0), input, False, True)
        self.head.set_keyfield(
            key_hash, 
            pow_nounce,
            key_height = 0,
            keyprblm_tx = genesis_prblm_tx)
        self.keyblocks.append(self.head)
        self.lastblock = self.head
    
    def get_keyblocks(self):
        '''获取链中所有的keyblock'''
        keyblocks:list[Block] = []
        q = [self.head]
        while q:
            block = q.pop(0)
            if block.iskeyblock:
                keyblocks.append(block)
            q.extend(block.next)
        return keyblocks
    
    def get_keyblocks_pref(self):
        '''获取链中所有的keyblock'''
        return self.keyblocks
    
    
    def get_feasible_keyblocks(self):
        """获取链中所有feasible的keyblock"""
        kbs = self.get_keyblocks_pref()
        if len(kbs) == 0:
            return []
        infeasi_kbs = []
        for kb in kbs:
            # print(kb.name, kb.keyfield.pre_key_feasible)
            if kb.keyfield.pre_key_feasible is False:
                infeasi_kbs.append(kb.keyfield.pre_kb)
        feasi_kbs = [kb for kb in kbs if (kb not in infeasi_kbs 
                    and kb.keyfield.key_tx is not None)]
        # print("feasi_kbs", [kb.name for kb in feasi_kbs])
        return feasi_kbs

    def get_mbs_after_kb(self, keyblock:Block) -> list[Block]:
        """
        获取某个keyblock下的所有miniblock
        """
        if not keyblock.iskeyblock:
            raise ValueError(f"Getting mbs: {keyblock.name} is not a keyblock")
        if keyblock.keyfield.key_tx is None:
            logger.warning("Getting mbs: key-tx of %s is None", keyblock.name)
            return []
        if len(keyblock.next) == 0:
            logger.warning("Getting mbs: %s have no further blocks", keyblock.name)
            return []
        q = [keyblock]
        miniblocks = []
        while q:
            block = q.pop(0)
            if not block.iskeyblock:
                miniblocks.append(block)
            if len(block.next) == 0:
                continue
            q.extend([b for b in block.next if not b.iskeyblock])
        return miniblocks

    def get_acpmbs_after_kb_and_label_forks(self, keyblock:Block) -> list[Block]:
        """
        获取某个keyblock下的所有被主链接受的miniblock
        """
        def select_acp_mbs(mbs:list[Block]):
            """如果有mini-block连接到"""
            pre_prblms = [mb.minifield.pre_p.pname for mb in mbs]
            fork_idxs = []
            for pre_p in pre_prblms:
                fork_idxs.append(tuple([x for x in range(len(pre_prblms)) 
                                        if pre_prblms[x] == pre_p]))
            acp_idxs = []
            # 随机选择重复的元素
            for idxs in list(set(fork_idxs)):
                if len(idxs) > 0:
                    acp_idxs.append(random.choice(idxs))
                else:
                    acp_idxs.append(idxs[0])
            acp_mbs = [mb for i,mb in enumerate(mbs) if i in acp_idxs]
            for mb in acp_mbs:
                mb.isFork = False
            return acp_mbs

        # 如果已经被记录了，直接返回结果
        if len(self.main_chain[keyblock.name]) > 0:
            logger.info(f"Miner {self.miner_id} accept miniblocks (record): "
                        f"{[b.name for b in self.main_chain[keyblock.name]]}")
            return self.main_chain[keyblock.name]
        
        if not keyblock.iskeyblock:
            raise ValueError(f"Getting acpmbs: {keyblock.name} is not a keyblock")
        if keyblock.keyfield.key_tx is None:
            logger.warning("Getting acpmbs: key-tx of %s is None", keyblock.name)
            return []
        if len(keyblock.next) == 0:
            logger.warning("Getting acpmbs: %s have no further blocks", keyblock.name)
            return []
        
        queue = [keyblock]
        acp_mbs = []
        while queue:
            block = queue.pop(0)
            if not block.iskeyblock:
                acp_mbs.append(block)
            if len(block.next) == 0:
                continue
            ftmd_mbs:list[Block] =  []
            for b in block.next:
                if not b.iskeyblock:
                    if b.minifield.bfthmd_state:
                        ftmd_mbs.append(b)
            if len(ftmd_mbs)==0:
                continue
            queue.extend(select_acp_mbs(ftmd_mbs))
        # 记录主链上所有的miniblock
        logger.info("Miner %s accept miniblocks: %s", self.miner_id, 
                    [b.name for b in acp_mbs])
        self.main_chain[keyblock.name].extend(acp_mbs)
        return acp_mbs

    def get_acpmbs_before_kb(self, keyblock:Block):
        """
        根据下一个keyblock选择的fathomed problem获取被接受的miniblock
        """
        mbs = self.get_mbs_after_kb(keyblock)
        if len(mbs) == 0:
            return []
        if len(keyblock.keyfield.accept_mbs) == 0:
            return []
        return [mb for mb in mbs if mb.name in keyblock.keyfield.accept_mbs]


    def search_forward(self, block: Block, searchdepth=500):
        '''Search a given block forward, that is, traverse the entire tree from a given search depth.
        Return the block if successfully searched, return None otherwise.
        '''
        if not self.head or not block:
            return None
        searchroot = self.lastblock
        if block.blockhead.height < searchroot.blockhead.height - searchdepth:
            return None  # 如果搜索的块高度太低 直接不搜了
        i = 0
        while searchroot and searchroot.pre and i <= searchdepth:
            if block.blockhead.blockhash == searchroot.blockhead.blockhash:
                return searchroot
            else:
                searchroot = searchroot.pre
                i = i + 1
        q = [searchroot]
        while q:
            blocktmp = q.pop(0)
            if block.blockhead.blockhash == blocktmp.blockhead.blockhash:
                return blocktmp
            for i in blocktmp.next:
                q.append(i)
        return None

    def search_chain_backward(self, block: Block, searchdepth=500):
        '''Search a given block backward from the lastblock using block hash, that is, ignore the forks.
        Return the block if successfully searched, return None otherwise.'''
        if not self.head:
            return None
        blocktmp = self.lastblock
        i = 0
        while blocktmp and i <= searchdepth:
            if block.blockhead.blockhash == blocktmp.blockhead.blockhash:
                return blocktmp
            blocktmp = blocktmp.pre
            i = i + 1
        return None
    
    def search_forward_by_hash(self, blockhash, searchdepth = 500):
        '''Search for a block with given hash forward. Traverse the entire tree from a given search depth.
        Return the block if successfully searched, return None otherwise.'''
        if not self.head:
            return None
        searchroot = self.lastblock
        i = 0
        while searchroot and searchroot.pre and i <= searchdepth:
            if blockhash == searchroot.blockhead.blockhash:
                return searchroot
            else:
                searchroot = searchroot.pre
                i = i + 1
        q = [searchroot]
        while q:
            blocktmp = q.pop(0)
            if blockhash == blocktmp.blockhead.blockhash:
                return blocktmp
            for i in blocktmp.next:
                q.append(i)
        return None

    def get_lastblock(self):  # 返回最深的block，空链返回None
        return self.lastblock

    def is_empty(self):
        if not self.head:
            print("Chain Is empty")
            return True
        else:
            return False

    def Popblock(self):
        popb = self.get_lastblock()
        last = popb.pre
        if not last:
            return None
        else:
            last.next.remove(popb)
            popb.pre = None
            return popb
        

    def inc_heavy_before(self, block:Block):
        """
        更新父区块的heavy，直到keyblock
        """
        if block.iskeyblock:
            return
        # logger.info(f"{self.miner_id}: updating heavy before {block.name}")
        pre_b = block
        while True:
            pre_b = pre_b.pre
            pre_b.inc_heavy()
            if pre_b.iskeyblock:
                break
            if pre_b.pre is None:
                break

    
    def add_block_direct(self, block: Block, pre: Block = None, next: Block = None):
        # 根据定位添加block，如果没有指定位置，加在最深的block后
        # 和AddChain的区别是这个是不拷贝直接连接的
        addSuccess = False

        # if self.search_forward(block):
        #     logger.warning(f"Chian{self.miner_id}: Block "
        #                    f"{block.name} is already included.")

        if not self.head:
            self.head = block
            self.lastblock = block
            addSuccess = True
            if not block.iskeyblock:
                return 
            self.keyblocks.append(block)
            logger.info(f"Miner{self.miner_id}: set head {block.name}, "
                        f"cur keyblocks:{[b.name for b in self.keyblocks]}")
            return
        
        # 指定前驱后继
        if pre and next:
            if not self.search_forward(pre) or not self.search_forward(next):
                logger.warning("Position Error")
                return addSuccess
            if next in pre.next:
                pre.next.append(block)
                block.pre = pre
                next.pre = block
                block.next.append(next)
                addSuccess = True
        # 指定前驱未指定后继
        elif pre and not next:
            # if not self.search_forward(pre):
            #     logger.warning("Position Error")
            #     return addSuccess
            pre.next.append(block)
            block.pre = pre
            addSuccess = True
        # 指定后继未指定前驱
        elif not pre and next:
            if not self.search_forward(next):
                print("Position Error")
                return addSuccess
            pre = next.pre
            pre.next.remove(next)
            pre.next.append(block)
            block.pre = pre
            block.next.append(next)
            next.pre = block
            addSuccess = True
        # 未指定，默认加在末尾
        elif not pre and not next:
            lastblock = self.get_lastblock()
            lastblock.next.append(block)
            block.pre = lastblock
            addSuccess = True
        
        if block.iskeyblock and addSuccess:
            self.keyblocks.append(block)
            logger.info(f"Miner{self.miner_id}: add keyblock {block.name}, "
                        f"length of keyblocks: {len(self.keyblocks)}, "
                        f"cur keyblocks:{[b.name for b in self.keyblocks]}")
        self.inc_heavy_before(block)
        self.lastblock = block
        return addSuccess

    def add_block_copy(self, newblock: Block):
        # 返回值：深拷贝插入完之后新插入链的块头
        if not newblock:  # 接受的链为空，直接返回
            return None
        copylist = []  # 需要拷贝过去的区块list
        local_block = self.search_chain_backward(newblock)
        # 把所有本地链没有的块都拷贝
        while newblock and not local_block:
            copylist.append(newblock)
            newblock = newblock.pre
            local_block = self.search_chain_backward(newblock)
        if local_block:
            while copylist:
                newblock = copylist.pop()
                copied_block = copy.deepcopy(newblock)
                copied_block.pre = local_block
                copied_block.next = []
                local_block.next.append(copied_block)
                local_block = copied_block
            #if lastblock.BlockHeight() > self.lastblock.BlockHeight():
            #    self.lastblock = lastblock  # 更新global chain的lastblock
        return local_block  # 返回深拷贝的最后一个区块的指针，如果没拷贝返回None
        


    def show_tree(self):
        # 打印树状结构
        queue = [self.head]
        printnum = 1
        while queue:
            length = 0
            print("|    ", end="")
            print("-|   " * (printnum - 1))
            while printnum > 0:
                queue.extend(queue[0].next)
                blockprint = queue.pop(0)
                length += len(blockprint.next)
                print("{}   ".format(blockprint.name), end="")
                printnum -= 1
            print("")
            printnum = length

    def show_chain_by_graphviz( self, 
        attack_record:dict = None, 
        success_record:dict = None,
        graph_path = None,
        graph_title = None,):
        '''借助Graphviz将区块链可视化'''
        def get_subprblm_label(pname:tuple):
            """获取子问题对的label"""
            subpair_name = pname[0]
            prblm_id = pname[1]
            if prblm_id > 0:
                return f'P{subpair_name}:<f1>'
            elif prblm_id < 0:
                return f'P{subpair_name}:<f2>'
            else:
                return f'P{subpair_name[0]}:<f0>'
    
        def set_miniblock_cluster(block:Block):
            """建立miniblock cluster"""
            with bc_graph.subgraph(name = f'cluster{block.name}',
                                   node_attr = {'shape': 'plaintext'}) as mb_cluster:
                # 设置集群参数
                cluster_color = 'lightgrey'if not block.isAdversary else '#fa8072'
                # 计算攻击成功概率
                atk_rate = (math.log(success_record[block.name]/attack_record[block.name]) 
                            if attack_record and block.name in attack_record else 0)
                theory_rate = block.minifield.atk_rate
                x_nk = block.minifield.subprblm_pairs[0][0].x_nk
                # 设置label
                label = ((f'''<
                    <TABLE border="0" cellborder="1" cellspacing="0">        
                        <TR><TD>{block.name} M{block.blockhead.miner_id}</TD></TR>
                        <TR><TD>r{block.blockhead.timestamp}, x_nk{x_nk}</TD></TR>
                        <TR><TD>heavy {block.get_heavy()}</TD></TR>
                        <TR><TD>fthmd_state {block.get_fthmstat()}</TD></TR>
                        <TR><TD><font color="red">theory_rate {theory_rate}</font></TD></TR>
                    </TABLE>>''')
                if attack_record is None else (f'''<
                    <TABLE border="0" cellborder="1" cellspacing="0">        
                        <TR><TD>{block.name} M{block.blockhead.miner_id}</TD></TR>
                        <TR><TD>r{block.blockhead.timestamp}, x_nk{x_nk}</TD></TR>
                        <TR><TD>fthmd_state {block.get_fthmstat()}</TD></TR>
                        <TR><TD><font color="red">success_rate {atk_rate}</font></TD></TR>
                        <TR><TD><font color="red">theory_rate {theory_rate}</font></TD></TR>
                    </TABLE>>'''))
                # label = ((f'''<
                #     <TABLE border="0" cellborder="1" cellspacing="0">        
                #         <TR><TD>{block.name} M{block.blockhead.miner_id} R{block.blockhead.timestamp}</TD></TR>
                #     </TABLE>>'''))
                mb_cluster.attr(style ='filled', color = cluster_color, label = label)
                # label_node = mb_cluster.node(block.name,
                #     label=(f'''<
                #     <TABLE border="0" cellborder="1" cellspacing="0">        
                #         <TR><TD>{block.name}</TD></TR>
                #         <TR><TD></TD></TR>
                #         <TR><TD>round {block.blockhead.timestamp}</TD></TR>
                #     </TABLE>>'''))
                # 建立问题对节点
                set_subpair_nodes(mb_cluster, block)
        
        def set_subpair_nodes(mb_cluster, block:Block):
            """建立问题对节点"""
            p1:LpPrblm
            p2:LpPrblm
            for (p1,p2) in block.minifield:
                pair_name = p1.pname[0]
                pair_str = f'P{pair_name}'
                pname1 = f'P<SUB>{pair_name[0]},{pair_name[1]}+</SUB>'
                pname2 = f'P<SUB>{pair_name[0]},{pair_name[1]}-</SUB>'
                color1 = 'red' if p1.all_integer() else 'black'
                color2 = 'red' if p2.all_integer() else 'black'
                color1 = '#77AC30' if not p1.feasible else color1
                color2 = '#77AC30' if not p2.feasible else color2
                color1 = 'blue' if p1.lb_prblm is not None else color1
                color2 = 'blue' if p2.lb_prblm is not None else color2
                zlp1 = round(p1.z_lp,2) if p1.z_lp else None
                zlp2 = round(p2.z_lp,2) if p2.z_lp else None
                mb_cluster.node(
                    pair_str,
                    
                    f'''<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                    <TR><TD PORT="f1"><font color="{color1}">{pname1}</font></TD>
                    <TD PORT="f2"><font color="{color2}">{pname2}</font></TD>
                    </TR></TABLE>>'''
                    )
                # f'''<
                    # <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                    #     <TR><TD PORT="f1"><font color="{color1}">{pname1}</font></TD></TR>
                    #     <TR><TD PORT="f2"><font color="{color2}">{pname2}</font></TD></TR>
                    # </TABLE>>'''
                    

        def set_keyblock_cluster(b:Block):
            """建立keyblock cluster"""
            key_pname = f'P{b.get_keyid()}' if b.get_keyid() is not None else 'None'
            with bc_graph.subgraph(
                name=f'cluster{b.name}', 
                node_attr={'shape': 'plaintext'}
            ) as c:
                label = (f'''<
                    <TABLE border="0" cellborder="0" cellspacing="0">        
                        <TR><TD>{b.name} M{b.get_miner_id()} </TD></TR>
                        <TR><TD>r{b.blockhead.timestamp}, heavy {b.get_heavy()}</TD></TR>
                        <TR><TD>fthmd_state {b.get_fthmstat()}</TD></TR>
                    </TABLE>>''')
                label = (f'''<
                    <TABLE border="0" cellborder="0" cellspacing="0">        
                        <TR><TD>{b.name} M{b.get_miner_id()} R{b.blockhead.timestamp}</TD></TR>
                    </TABLE>>''')
                c.attr(label = label)
                node_name = key_pname if key_pname != 'None' else b.name
                color = 'red' if b.get_keyprblm() is not None and b.get_keyprblm().all_integer() else 'black'
                c.node(node_name, f'''<
                        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                            <TR><TD PORT="f0"><font color="{color}">{key_pname}</font></TD></TR>
                        </TABLE>>''')
        
        def set_subprblm_edges(block:Block):
            """建立子问题对和前一问题的连接"""
            p1:LpPrblm
            p2:LpPrblm
            for (p1,p2) in block.minifield:
                if p1.pre_pname[1] != 0:
                    subnode_label = f'P{p1.pname[0]}'
                    preprblm_label = get_subprblm_label(p1.pre_pname)
                    bc_graph.edge(
                        preprblm_label, 
                        subnode_label, 
                        f"x_nk {p1.x_nk}\nrest_x {p1.pre_rest_x}", 
                        fontcolor = "blue")
                else:
                    subnode_label = f'P{p1.pname[0]}'
                    # preprblm_label = f'{block.pre.name}:f0'
                    preprblm_label = get_subprblm_label(p1.pre_pname)
                    bc_graph.edge(
                        preprblm_label, 
                        subnode_label, )
                        # f"x_nk {p1.x_nk}",
                        # fontcolor = "blue" )
                # if p1.lb_prblm is not None:
                #     # print(get_subprblm_label(p1.pname), get_subprblm_label(p1.lb_prblm.pname))
                #     bc_graph.edge(get_subprblm_label(p1.pname), 
                #                   get_subprblm_label(p1.lb_prblm.pname),
                #                   arrowsize='0.5', color='blue',constraint='false')
                # if p2.lb_prblm is not None:
                #     # print(get_subprblm_label(p2.pname), get_subprblm_label(p2.lb_prblm.pname))
                #     bc_graph.edge(get_subprblm_label(p2.pname), 
                #                   get_subprblm_label(p2.lb_prblm.pname) , 
                #                   arrowsize='0.5', color='blue',constraint='false')
            
        def set_keyprblm_edges(block:Block):
            """建立keyprblm和前一子问题的连接"""
            key_edge_color = 'red' if len(block.keyfield.opt_prblms)>0 else 'black'
            presub_label = None
            if block.keyfield.pre_pname is not None:
                presub_label = get_subprblm_label(block.keyfield.pre_pname)
                bc_graph.edge(presub_label, f"{block.name}:f0", color=key_edge_color)
            # for p in block.keyfield.fthmd_prblms:
            #     #     # print(get_prblm_label(p.pname)
            #     if get_subprblm_label(p.pname) == presub_label:
            #         continue
                # bc_graph.edge(get_subprblm_label(p.pname), f"{block.name}:f0",  
                #                 arrowsize='0.5', style='dashed', color='grey',constraint='false')

        
        bc_graph = graphviz.Digraph('Blockchain Structure', engine='dot', 
                                    node_attr={'shape': 'plaintext'})
        # bc_graph.attr(rankdir='LR')
        q = [self.head]
        while q:
            block = q.pop(0)
            if not block.iskeyblock:
                # 建立miniblock集群和其中的subprblm node
                set_miniblock_cluster(block)
                # 建立edges
                set_subprblm_edges(block)
            elif block.iskeyblock:
                # 建立miniblock集群和其中的subprblm node
                set_keyblock_cluster(block)
                # 建立edges
                set_keyprblm_edges(block)
                # for p in block.keyfield.fthmd_prblms:
                #     # print(get_prblm_label(p.pname)
                #     bc_graph.edge(get_prblm_label(p.pname), f'{orig_pname}')

            if block.next:
                q.extend(block.next)
        # 生成矢量图,展示结果
        g_title = graph_title if graph_title is not None else "blockchain_visualization"
        g_path =graph_path if graph_path is not None else self.background.get_result_path()
        bc_graph.render(directory=g_path / g_title, format='svg', view=False)

    
    def printchain2txt(self, chain_data_path=None, file_name=None):
        '''
        前向遍历打印链中所有块到文件
        param:
            blockchain
            chain_data_url:打印文件位置,默认'chain_data.txt'
        '''
        def save_chain_structure(chain,f):
            blocklist = [chain.head]
            printnum = 1
            while blocklist:
                length = 0
                print("|    ", end="",file=f)
                print("-|   " * (printnum - 1),file=f)
                while printnum > 0:
                    blocklist.extend(blocklist[0].next)
                    blockprint = blocklist.pop(0)
                    length += len(blockprint.next)
                    print("{}   ".format(blockprint.name), end="",file=f)
                    printnum -= 1
                print("",file=f)
                printnum = length

        if chain_data_path is None:
            chain_data_path = self.background.get_chain_data_path()
        if file_name is None:
            file_name = f"chain_data{self.miner_id}.txt"
        
        if not os.path.exists(chain_data_path):
            chain_data_path.mkdir(parents=True)
        
        if not self.head:
            with open(chain_data_path / file_name,'w+') as f:
                print("empty chain",file=f)
            return
        
        with open(chain_data_path / file_name, 'w+') as f:
            print("Blockchian maintained BY Miner",self.miner_id,file=f)    
            # 打印主链
            save_chain_structure(self,f)
            #打印链信息
            q:list[Block] = [self.head]
            blocklist = []
            while q:
                block = q.pop(0)
                blocklist.append(block)
                print(block, file=f)
                for i in block.next:
                    q.append(i)
    
if __name__ == "__main__":
    
    import time

    import graphviz
    
    s = graphviz.Digraph('structs', filename='structs.gv',
                        node_attr={'shape': 'plaintext'})

    s.node('struct1', '''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR>
        <TD>left</TD>
        <TD PORT="f1"><font color="red">middle</font></TD>
        <TD PORT="f2">right</TD>
    </TR>
    </TABLE>>''')
    color1 = "red"
    color2 = "black"
    subpair_name = (1,2)
    s.node('struct2',
                    '''<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR>
        <TD COLSPAN="4">&#x2119;<SUB>h</SUB></TD>
    </TR>
    <TR>
        <TD >c</TD>
        <TD COLSPAN="2" PORT="f1">[15, -8]</TD>
        <TD PORT="f2">h<sub>ub</sub></TD>
    </TR>
    <TR>
    <TD ROWSPAN="3">G<sub>ub</sub></TD>
        <TD COLSPAN="2">[1,2]</TD>
        <TD>1</TD>
    </TR>
    <TR>
        <TD COLSPAN="2">[1,2]</TD>
        <TD>2</TD>
    </TR>
    <TR>
        <TD COLSPAN="2">[1,2]</TD>
        <TD>3</TD>
    </TR>
    <TR>
        <TD PORT="f7">x<sub>lp</sub></TD>
        <TD PORT="f8">[0,1]</TD>
        <TD PORT="f9">z<sub>lp</sub></TD>
        <TD >1.5</TD>
    </TR>
</TABLE>>''')
    s.node('struct3', '''<
    <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    <TR>
        <TD PORT="here" ROWSPAN="3">hello<BR/>world</TD>
        <TD COLSPAN="3">b</TD>
        <TD ROWSPAN="3">g</TD>
        <TD ROWSPAN="3">h</TD>
    </TR>
    <TR>
        <TD>c</TD>
        <TD >d</TD>
        <TD>e</TD>
    </TR>
    <TR>
        <TD COLSPAN="3">f</TD>
    </TR>
    </TABLE>>''')

    s.edges([('struct1:f1', 'struct2:f1'), ('struct1:f2', 'struct3:here')])

    s.view()