from miner.miner import Miner
from data import NewBlocks
from .network_abc import Network

class SynchronousNetwork(Network):
    """同步网络"""

    def __init__(self, global_var, miners: list[Miner]):
        super().__init__(global_var)
        self.miners = miners
        self.network_tape:list[NewBlocks] = []  # network_tape存储要广播的新块和对应的信息
        # 元素chain_packet为字典存储新块及'minerid'、'TTL'等信息

        # with open(self.NET_RESULT_PATH / 'network_log.txt', 'a') as f:
        #     print('Network Type: FullConnectedNetwork', file=f)

    def set_net_param(self):
        pass

    def access_network(self, newblock:NewBlocks, minerid, round):
        """ 本轮新产生的块添加到network_tape\n
        param: newblock type:block
               minerid type:int
        """
        block_packet = {
            'minerid': minerid,
            'blocks': newblock
        }
        self.network_tape.append(block_packet)

    def clear_NetworkTape(self):
        """清空network_tape"""
        self.network_tape = []

    def diffuse(self, round):
        """全连接无延迟广播network_tape中的块\n
        param: miners 全部矿工 type:miner
        """
        if self.network_tape:
            for miner in self.miners:
                for block_packet in self.network_tape:
                    if miner.miner_id != block_packet['minerid']:  # 添加到miner的receive_tape中
                        miner.receive_blocks(block_packet['blocks'],round)
            self.clear_NetworkTape()