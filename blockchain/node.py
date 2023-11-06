from kademlia.network import Server
from kademlia.crawling import ValueSpiderCrawl
from block.transaction import Transaction
from block.transaction_pool import Transaction_pool
from block.block import Block
from block.blocklist import BlockList
from block.blockchain import BlockChain
from protocol import DataExchangeProtocol
from utils.utils import *
import json
import math
import random
import logging
import nest_asyncio
import asyncio

log = logging.getLogger("Blockchain")
nest_asyncio.apply()


class Node(Server):
    def __init__(self, ksize=20, alpha=3, node_id=None, storage=None, coin=0):
        super().__init__(ksize, alpha, node_id.encode('utf8'), storage)
        self.psu_key = node_id
        self.trans_pool : Transaction_pool = Transaction_pool()
        self.blockchain : BlockChain = BlockChain()
        self.protocol_class = DataExchangeProtocol

    def verify_trans(self, trans : Transaction):
        r"""Verifies the formate of the transaction.
        Args:
            trans[Transaction]: transaction want to verify
        Returns:
            bool
        """
        return trans.hash == trans.__digest__()
    
    def verify_block(self, block : Block):
        r"""Verifies the all transaction contained in the block.
        """
        res = list(map(self.verify_trans, block.trans))
        return False not in res
    
    def accept_block(self, block : Block):
        r'''Add a block into node's blockchain.
        This function will remove all transaction contained in the block.
        '''
        for trs in block.trans:
            if trs in self.trans_pool: self.trans_pool.remove(trs)
        self.blockchain.add_block(block)

    def _create_protocol(self):
        return self.protocol_class(self.node, self.storage, self.ksize, self)


    async def listen(self, port, interface="0.0.0.0"):
        r"""Listen local UDP address of node.
        This address will be a port that accepts message. 
        """
        self.node.ip = interface; self.node.port = port
        loop = asyncio.get_event_loop()
        listen = loop.create_datagram_endpoint(self._create_protocol, local_addr=(interface, port))
        log.info("Node %i listening on %s:%i",
            self.node.long_id, interface, port)
        self.transeport, self.protocol = await listen
        self.refresh_table()
        self.syncronize()

    def neighbors(self) -> list:
        r"""Get neighbor node.
        Returns:
            Kademlia.Node
        """
        return self.protocol.router.find_neighbors(self.node)

    async def broadcast(self, call_method, message, exclude : list=[]):
        r"""Broadcast a message to neighbor node.
        Args:
            call_method [Protocol.call__]: a call function which be used to call remote function.
            message [str]: a message want to pass
            exclude [list[str]]: some node that will be exclude from broadcast node.
        """
        neighbors = self.neighbors()
        if exclude:
            neighbors = [neib for neib in neighbors if (neib.ip, neib.port) not in exclude]
            if len(neighbors) == 0: return
        cos = [call_method((nb.ip, nb.port), message) for nb in neighbors]
        gathered = await asyncio.gather(*cos)

    def make_transaction(self, to, data):
        r"""makes a transaction for node.
        Notes:
            the method will return a transaction that source is always this node.
        Args:
            to [str]: the receiver long_id of this transaction.
            data [any]: the coin in this version.
        Returns:
            Transaction
        """
        data_hash = data_digest(data)
        trans = Transaction.Transe_Create(self.psu_key, to, data_hash)
        asyncio.run(self.set(data_hash, data))
        return trans


    async def broadcast_transaction(self, trans : Transaction):
        r"""Broadcast the tx to neighbors and the tx will not be broadcast if the tx can not be 
        verified successfully in local.
        """
        if self.verify_trans(trans):
            self.trans_pool.append(trans)
        await self.broadcast(self.protocol.call_forward_trans, str(trans))

    def make_block(self, trans_pool : Transaction_pool):
        r"""Creates a block with provided transactions list.
        Args:
            trans_pool [Transaction_pool]: transaction list.
        Returns:
            new block [Block].
        """
        if len(trans_pool) == 0: raise RuntimeError("You can not create a empty block.")
        pre_block = self.blockchain.last_block()
        return Block.Create_Block(trans_pool, 0 if not pre_block else pre_block.id+1, 
                                  self.psu_key, None if not pre_block else pre_block.hash, None)

    
    async def broadcast_block(self, block):
        r"""Broadcasts block to neighbors and the block will not be broadcast if it can not be 
        verified succesfully.
        """
        if self.verify_block(block):
            self.accept_block(block)
        await self.broadcast(self.protocol.call_forward_block, str(block))


    async def node_bootstrap(self, boot_node_addr):
        r"""bootstrap a new node with provided node address list. The provided node will help
        to enter blockchain network.
        Args:
            boot_node_addr [list[tuple[str, int]]]: a bootstrap address list.
        """
        nearist = await self.bootstrap(boot_node_addr)
        if (len(nearist) == 0):
            log.error("Faile to bootstrap due to invalid bootable node address.")
            return
        response = await self.protocol.call_get_chain_state(boot_node_addr[0])
        self.blockchain = BlockChain.Blockchain_Decode(response[0])
        self.trans_pool = Transaction_pool.TransPool_Decode(response[1])
        log.info("Successfully to init blockchian status.")
        return True
    
    def syncronize(self, interval=1800):
        r"""Syncronizes node status containing chain and tx_pool within a interval time.
        Args:
            interval [int]: syncronize time.
        """
        loop = asyncio.get_event_loop()
        neighbors = self.neighbors()
        if len(neighbors) == 0:
            log.info("There is no beighbors to syncronize.")
        else:
            choice_size = int(math.sqrt(len(neighbors))) if len(neighbors) > 1 else 1
            neighbors = random.sample(neighbors, choice_size)
            loop.run_until_complete(self.__syncronize__(neighbors))
        self.syncronize_loop = loop.call_later(interval, self.syncronize)

    async def __syncronize__(self, neighbors:list):
        r"""The syncronize function. The prosses is:
        1. gets all digest of blockchain and transaction pool from all neighbors.
        2. if there are any block or transaction that the node didont, send a request to
        3. get the block or tx and add it.
        """
        task = [self.protocol.call_get_blocks_digest((n.ip, n.port)) for n in neighbors]
        resp = await asyncio.gather(*task)
        task = [self.__syncronize_block__(d) for d in resp]
        await asyncio.gather(*task)

        task = [self.protocol.call_get_txs_digest((n.ip, n.port)) for n in neighbors]
        resp = await asyncio.gather(*task)
        task = [self.__syncronize_tx__(d) for d in resp]
        await asyncio.gather(*task)


    async def __syncronize_tx__(self, tx_digest):
        if not tx_digest or tx_digest["hashs"]: return
        dig_list = [h for h in tx_digest["hashs"] if not self.hastx(h)]
        if len(dig_list) == 0: return
        resp = await self.protocol.call_get_transactions((tx_digest["addr"][0], tx_digest["addr"][1]), dig_list)
        if not resp: return
        for tx in resp:
            if self.verify_trans(tx): self.add_transaction(tx)
        log.info("Succesfully syncronizes {} tx.".format(len(resp)))

    async def __syncronize_block__(self, block_digest):
        if not block_digest or not block_digest["hashs"]: return 
        dig_list = [h for h in block_digest["hashs"] if not self.hasbk(h)]
        if len(dig_list) == 0: return
        resp = await self.protocol.call_get_blocks((block_digest["addr"][0], block_digest["addr"][1]), dig_list)
        if not resp: return
        for block in resp:
            if self.verify_block(block): self.accept_block(block)
        log.info("Succesfully syncronizes {} blocks.".format(len(resp)))
    
    def hastx(self, tx_hash : str) -> bool:
        return self.trans_pool.ishas(tx_hash)
    
    def hasbk(self, bk_hash : str) -> bool:
        return self.blockchain.ishas(bk_hash)
        
    def get_tx_digest(self):
        return self.trans_pool.get_message_digest()
    
    def get_block_digest(self):
        return self.blockchain.get_message_digest()
        
    def find_tx(self, tx : str) -> Transaction:
        return self.trans_pool.find(tx)
        
    def find_block(self, block : str) -> Block:
        return self.blockchain.find(block)
    
    def add_transaction(self, tx : Transaction):
        self.trans_pool.append(tx)

    def get_value(self, value_hash :str):
        print(asyncio.run(self.get(value_hash)))

    def print_block_header(self):
        b0 = self.blockchain.creater_block()
        if not b0: print("There are not block in local blockchain."); return
        while True:
            print(b0.header())
            if b0.next_hash == None: break
            b0 = self.find_block(b0.next_hash)

    def print_block(self, block_hash : str):
        block = self.find_block(block_hash)
        if not block:
            print("no such block:{}".format(block_hash))
            return
        print("header:")
        print(block.header())
        print("body:")
        print(block.body())

    def print_transaction_pool(self):
        print(self.trans_pool.__list__())