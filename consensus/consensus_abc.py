from abc import ABCMeta, abstractmethod

from chain import Block, Chain

class Consensus(metaclass=ABCMeta):        #抽象类

    @abstractmethod
    def setparam(self):
        '''设置共识所需参数'''
        pass

    @abstractmethod
    def mining_consensus(self, blockchain:Chain, miner_id, 
                    is_adversary = None, input = None, q = None, round = None):
        '''共识机制定义的挖矿算法
        :param blockchain: 当前区块链
        :param miner_id: 矿工编号
        :param is_adversary: 是否是攻击者
        :param input: 要写入区块的内容
        :param q: 进行哈希计算的次数（PoW）
        :param round: 当前轮次
        return:
            新产生的区块  type:Block 
            挖矿成功标识    type:bool
        '''
        newblock:Block = Block()
        mining_success:bool = bool()
        return newblock, mining_success

    @abstractmethod
    def valid_chain(self):
        '''检验链是否合法
        return:
            合法标识    type:bool
        '''
        pass

    @abstractmethod
    def valid_block(self):
        '''检验单个区块是否合法
        return:合法标识    type:bool
        '''
        pass