import copy
import logging
import math
import os
import random
from collections import defaultdict

import graphviz

from background import Background
from .txpool import Transaction
from .block import Block
from .blockhead import BlockHead
from .lpprblm import LpPrblm

logger = logging.getLogger(__name__)


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