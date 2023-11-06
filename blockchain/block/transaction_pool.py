from block.transaction import Transaction
from utils.utils import *
from copy import deepcopy
import json


class Transaction_pool(list):

    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()


    def append(self, __object: Transaction) -> None:
        if __object not in self:
            return super().append(__object)
        
    def remove(self, __value: Transaction) -> None:
        return super().remove(__value)
    
    def find(self, hash : str):
        hash_list = [h.hash for h in self]
        if self.ishas(hash):
            return self[hash_list.index(hash)]
        return None
    
    def get_message_digest(self):
        return {"len": len(self), "hashs": [h.hash for h in self]}

    
    def ishas(self, trans_hash : str):
        return trans_hash in [t.hash for t in self]
    
    def return_all(self) -> list:
        return list(self)
    
    
    @staticmethod
    def TransPool_Decode(string : str):
        trans_list = json.loads(string)
        pool = Transaction_pool()
        for trans_dict in trans_list:
            pool.append(Transaction.Trans_Decode(trans_dict))
        return pool
    
    def __list__(self) -> list:
        return [deepcopy(tx.__dict__) for tx in self]
    
    def __str__(self) -> str:
        return json.dumps([str(ts) for ts in self]).replace('\'', '\"')
    
    def __digest__(self) -> str:
        if len(self) == 0:
            raise Exception('The list of transactions is empty')
        bts = self[0].hash
        for ts in self[1:]:
            bts += ts.hash
        return digest(bts).hex()
    