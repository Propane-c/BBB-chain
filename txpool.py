import copy
import lpprblm
class Transaction(object):
    def __init__(self, tx_from = None, tx_nonce = None, data = None) -> None:
        '''In branchbound, data is a origin key problem'''
        self.tx_from = tx_from
        self.tx_nonce = tx_nonce
        self.data:lpprblm.LpPrblm = data

    def __eq__(self, other: 'Transaction'):
        if (other.tx_from == self.tx_from and
            other.tx_nonce == self.tx_nonce):
            # if other.tx_data == self.tx_data:
            return True
            # else:
            #     raise ValueError(f"The transaction from {self.tx_from} with nonce"
            #             f" {self.tx_nonce} have two different data!")
        else:
            return False

    def __repr__(self):
        txlist = []
        for k, v in self.__dict__.items():
            txlist.append(k + ': ' + str(v))
        return '\n'.join(txlist)


class TxPool(object):
    def __init__(self) -> None:
        # self.pending: list[Transaction] = [Transaction('env', 1, lpprblm.test1())]
        # self.pending: list[Transaction] = [Transaction('env', 1, lpprblm.prblm_generator())]
        self.pending: list[Transaction] = []
        self.queued: list[Transaction] = []
        # for prblm_num in range(5-2):# 预生成问题
            # self.queued.append(Transaction('env', prblm_num+2, lpprblm.test1()))
            # self.queued.append(Transaction('env', prblm_num + 2 , lpprblm.prblm_generator(20)))

    def load_prblm_pool(self,  prblm_pool:list[lpprblm.LpPrblm]):
        """载入问题池"""
        for prblm_index, prblm in enumerate(copy.deepcopy(prblm_pool)):
            self.queued.append(Transaction('env', prblm_index + 1 , prblm))
        self.pending.append(self.queued.pop(0))

    def reorg(self, rcv_tx: Transaction = None):
        """reorg the txpool

        Param:
            rcv_tx (Transaction): the received transaction
        """
        if rcv_tx is not None:
            self.pending = self.discard_tx(rcv_tx, self.pending)
            self.queued = self.discard_tx(rcv_tx, self.queued)
        if len(self.pending) == 0 and len(self.queued) != 0:
            self.pending.append(self.queued.pop(0))

    def discard_tx(self, rcv_tx: Transaction, tx_list: list[Transaction]):
        '''If the tx with same tx_form and tx_nonce already exits, 
        discard them from txpool
        
        :param rcv_tx: the received tx
        :param tx_list: pending or queued
        :return the reorged tx_list
        '''
        del_txs = []
        for tx_idx, tx in enumerate(tx_list):
            if (tx.tx_from == rcv_tx.tx_from and
                    tx.tx_nonce == rcv_tx.tx_nonce):
                del_txs.append(tx_idx)
        if del_txs:
            tx_list = [ptx for ptx_idx, ptx in enumerate(tx_list)
                       if ptx_idx not in del_txs]
        return tx_list

    def add_tx(self):
        pass