import copy
import logging
from dataclasses import dataclass

from functions import hashG, hashH

from .blockhead import BlockHead
from .lpprblm import LpPrblm
from .txpool import Transaction

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