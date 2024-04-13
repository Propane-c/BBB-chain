import copy

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