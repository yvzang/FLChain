from block.block import Block
from block.transaction_pool import Transaction_pool
from block.transaction import Transaction
import json
from copy import deepcopy

class BlockList(list):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
    
    def get_message_digest(self):
        return {"len": len(self), "hashs": [h.hash for h in self] if len(self) > 0 else None}
    
    def __list__(self) -> list:
        return [deepcopy(bk.__dict__) for bk in self]
    
    def __str__(self) -> str:
        return json.dumps([str(b) for b in self]).replace('\'', '\"')
    
    @staticmethod
    def Decode(list_str):
        lst = json.loads(list_str)
        return [Block.Block_Decode(b) for b in lst]
    