from block.transaction import Transaction
from block.transaction_pool import Transaction_pool
from copy import deepcopy
import time
import json

from utils.utils import *

block_reward=5

class Block:
    def __init__(self) -> None:
        self.trans : Transaction_pool = None
        self.trans_num = None
        self.id = None
        self.stamp = None
        self.creater = None
        self.pre_hash = None
        self.hash = None
        self.next_hash = None

    def header(self):
        attrs = deepcopy(self.__dict__)
        attrs.pop('trans')
        return attrs
    
    def body(self):
        return self.trans.__list__()

    @staticmethod
    def Create_Block(trans_pool, id, creater, pre_hash, next_hash):
        new_block = Block()
        new_block.trans : Transaction_pool = Transaction_pool(trans_pool.return_all())
        new_block.trans_num = len(new_block.trans)
        new_block.id = id
        new_block.creater = creater
        new_block.pre_hash = pre_hash
        new_block.next_hash = next_hash
        new_block.stamp = time.time()
        new_block.hash = new_block.__digest__()
        return new_block


    @staticmethod
    def Block_Decode(String : str):
        block_dict = json.loads(String)
        new_block = Block()
        for attr, value in block_dict.items():
            if not hasattr(new_block, attr):
                raise AttributeError("Invalid block message.")
            setattr(new_block, attr, value)
        #transaction message decoding
        new_block.trans = Transaction_pool.TransPool_Decode(new_block.trans)
        return new_block


    def __str__(self) -> str:
        dicts = deepcopy(self.__dict__)
        dicts['trans'] = str(self.trans)
        return json.dumps(dicts).replace('\'', '\"')
    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Block):
            return self.hash == __value.hash
        return False
    
    def __digest__(self) -> str:
        return digest(self.trans.__digest__() + digest(self.id).hex()+ digest(self.stamp).hex()).hex()
    