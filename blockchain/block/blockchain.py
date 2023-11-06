from block.block import Block
from block.blocklist import BlockList
from block.block import Transaction_pool
from block.transaction import Transaction
import json


class BlockChain:
    def __init__(self) -> None:
        self.blocks = BlockList()
        self.indexs = {}

    def creater_block(self) -> Block:
        if len(self.blocks) > 0:
            return self.blocks[0]
        return None

    def get_message_digest(self):
        return self.blocks.get_message_digest()

    def ishas(self, hash):
        return hash in [b.hash for b in self.blocks]

    def find(self, hash : str):
        if hash not in self.indexs.keys():
            return None
        return self.blocks[self.indexs[hash]]
    
    def get_UTXO(self, id):
        utxo = 0
        for block in self.blocks:
            for trans in block.trans:
                if id == trans.trans_from: utxo -= trans.data
                elif id == trans.to : utxo += trans.data
        return utxo


    def add_block(self, obj : Block):
        if self.ishas(obj.hash): return
        if obj.pre_hash:
            self.blocks[self.indexs[obj.pre_hash]].next_hash = obj.hash
        self.blocks.append(obj)
        self.indexs[obj.hash] = self.blocks.index(obj)
        return True

    def last_block(self) -> Block:
        r'''Get the Lastest block from chain.
        '''
        if len(self.blocks) == 0:
            return None
        return self.blocks[-1]
    
    @staticmethod
    def Blockchain_Decode(string : str):
        chain_dict = json.loads(string)
        chain = BlockChain()
        for block_str in chain_dict:
            block = Block.Block_Decode(block_str)
            chain.add_block(block)
        return chain

    
    def __str__(self) -> str:
        return str(self.blocks)
    
    def __len__(self) -> int:
        return len(self.blocks)