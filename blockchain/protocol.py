import asyncio
import logging
import json
from kademlia.protocol import KademliaProtocol
from block.blockchain import BlockChain
from block.blocklist import BlockList
from block.block import Block
from block.transaction import Transaction
from block.transaction_pool import Transaction_pool
from utils.utils import *

log = logging.getLogger("Blockchain")

class DataExchangeProtocol(KademliaProtocol):

    def __init__(self, source_node, storage, ksize, node):
        super().__init__(source_node, storage, ksize)
        self.node = node
        self.temp_data = {}

    async def rpc_transp_metadata(self, sender : tuple, senderid : str, metadata : str, digest : str):
        if not data_digest(metadata) == digest: return False
        self.temp_data[senderid] += metadata
        return True
    
    async def rpc_receive_trans(self, sender : tuple, senderid : str) -> bool:
        trans_str = self.temp_data[senderid]
        self.temp_data.pop(senderid)
        trans = Transaction.Trans_Decode(trans_str)
        if self.node.verify_trans(trans):
            self.node.add_transaction(trans)
            log.info("Succesfully received a new transaction.")


    async def rpc_forward_trans(self, sender, trans_str):
        trans = Transaction.Trans_Decode(trans_str)
        if not trans:
            log.error("There is a invalid json string of Transaction received.")
            return False
        if not self.node.hastx(trans.hash):
            if self.node.verify_trans(trans):
                log.info("Receive transaction digest from and forward to other node.")
                self.node.add_transaction(trans)
                task = await self.node.broadcast(self.call_forward_trans, trans_str, [sender])

    async def rpc_forward_block(self, sender, block_str):
        block = Block.Block_Decode(block_str)
        if not block:
            log.info("There is a invalid json string of Block received.")
            return False
        if not self.node.hasbk(block.hash):
            if self.node.verify_block(block):
                log.info("Receive block message from {} and forward to other node.")
                self.node.accept_block(block)
                task = await self.node.broadcast(self.call_forward_block, block_str, [sender])

    def rpc_get_transactions(self, sender, tx_hash : str):
        log.info("Receive a request for transactions detail from {}:{}".format(sender[0], sender[1]))
        tx_hash = json.loads(tx_hash)
        find_res = Transaction_pool(map(self.node.find_tx, tx_hash))
        find_res = Transaction_pool([r for r in find_res if r])
        return str(find_res)
    
    def rpc_get_tx_digest(self, sender):
        log.info("Receive a request for tx_digest from {}:{}".format(sender[0], sender[1]))
        digest = self.node.get_tx_digest()
        digest["addr"] = [self.node.node.ip, self.node.node.port]; digest["id"] = self.node.psu_key
        return json.dumps(digest)
    
    def rpc_get_blocks(self, sender, block_hash : str):
        log.info("Receive a request for blocks detail from {}:{}".format(sender[0], sender[1]))
        block_hash = json.loads(block_hash)
        find_res = BlockList(map(self.node.find_block, block_hash))
        find_res = BlockList([r for r in find_res if r])
        return str(find_res)
    
    def rpc_get_block_digest(self, sender):
        log.info("Receive a request for block_digest from {}:{}".format(sender[0], sender[1]))
        digest = self.node.get_block_digest()
        digest["addr"] = [self.node.node.ip, self.node.node.port]; digest["id"] = self.node.psu_key
        return json.dumps(digest)


    def rpc_get_chain_state(self, sender):
        log.info("Receive a request for status of chain from {}:{}".format(sender[0], sender[1]))
        return [str(self.node.blockchain), str(self.node.trans_pool)]
    
    
    async def call_get_chain_state(self, sender):
        log.info("Send a request for state of chain from {}:{}".format(sender[0], sender[1]))
        resp = await self.get_chain_state(sender)
        if not resp[0]:
            log.info("Faile to get blockchain state from bootstrap address due to network connection.")
            return None
        return resp[1]

    
    async def call_forward_trans(self, sender, trans_str):
        log.info("Send a transaction message to {}:{}".format(sender[0], sender[1]))
        return await self.forward_trans(sender, trans_str)
    
    async def call_forward_block(self, sender, block_str):
        log.info("Send a block message to {}:{}".format(sender[0], sender[1]))
        return await self.forward_block(sender, block_str)
    
    async def call_get_transactions(self, ask, tx_hashs : list):
        log.info("Send a transaction request message to {}:{}".format(ask[0], ask[1]))
        resp = await self.get_transactions(ask, json.dumps(tx_hashs))
        if not resp[0]:
            log.info("Fails to get transaction list from neighbors due to network.")
            return None
        return Transaction_pool.TransPool_Decode(resp[1])
    
    async def call_get_txs_digest(self, ask):
        log.info("Send a tx_digest request message to {}:{}".format(ask[0], ask[1]))
        resp = await self.get_tx_digest(ask)
        if not resp[0]:
            log.info("Fails to get a transaction digest due to network.")
            return None
        return json.loads(resp[1])
    
    async def call_get_blocks(self, ask, block_hashs : list):
        log.info("Send a block request message to {}:{}".format(ask[0], ask[1]))
        resp = await self.get_blocks(ask, json.dumps(block_hashs))
        if not resp[0]:
            log.info("Fails to get block list due to network.")
            return None
        return BlockList.Decode(resp[1])
    
    async def call_get_blocks_digest(self, ask):
        log.info("Send a block_digest request message to {}:{}".format(ask[0], ask[1]))
        resp = await self.get_block_digest(ask)
        if not resp[0]:
            log.info("Fails to get a block digest due to network.")
            return None
        return json.loads(resp[1])

