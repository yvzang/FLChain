import random
import asyncio
import logging

from rpcudp.protocol import RPCProtocol
import socket

from blockchain.utils.utils import *
from blockchain.kademlia.node import Node
from blockchain.kademlia.routing import RoutingTable
from blockchain.kademlia.utils import digest

log = logging.getLogger("Blockchain")  # pylint: disable=invalid-name


class KademliaProtocol(RPCProtocol):
    def __init__(self, source_node, storage, ksize):
        RPCProtocol.__init__(self)
        self.router = RoutingTable(self, ksize, source_node)
        self.storage = storage
        self.source_node = source_node
        self.temp_data = {}

    def get_refresh_ids(self):
        """
        Get ids to search for to keep old buckets up to date.
        """
        ids = []
        for bucket in self.router.lonely_buckets():
            rid = random.randint(*bucket.range).to_bytes(20, byteorder='big')
            ids.append(rid)
        return ids


    def rpc_stun(self, sender):  # pylint: disable=no-self-use
        return sender

    def rpc_ping(self, sender, nodeid):
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        return self.source_node.id
    
    def rpc_send_prepar(self, sender, datasize):
        key = random.random()
        receiv_addr = ("127.0.0.1", random.randint(2000, 8888))
        receiver = DataChenelReceiver(receiv_addr[1], datasize=datasize, host=receiv_addr[0])
        receiver.run()
        self.temp_data[key] = receiver
        return {'ip': receiv_addr[0], 'port': receiv_addr[1], 'key': key}


    def rpc_store(self, sender, nodeid, key, link_key):
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        log.debug("got a store request from %s, storing '%s'",
                  sender, key.hex())
        self.storage[key] = self.temp_data[link_key].result().decode()
        self.temp_data.pop(link_key)
        return True

    def rpc_find_node(self, sender, nodeid, key):
        log.info("finding neighbors of %i in local table",
                 int(nodeid.hex(), 16))
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        node = Node(key)
        neighbors = self.router.find_neighbors(node, exclude=source)
        return list(map(tuple, neighbors))

    def rpc_find_value(self, sender, nodeid, key):
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        value = self.storage.get(key, None)
        if value is None:
            return self.rpc_find_node(sender, nodeid, key)
        return self.call_send_data(sender, value)

    async def call_find_node(self, node_to_ask, node_to_find):
        address = (node_to_ask.ip, node_to_ask.port)
        result = await self.remote_rpc_packager(self.find_node, args=(address, self.source_node.id, node_to_find.id))
        return self.handle_call_response(result, node_to_ask)

    async def call_find_value(self, node_to_ask, node_to_find):
        address = (node_to_ask.ip, node_to_ask.port)
        log.debug("try to find value from {}:{}".format(address[0], address[1]))
        result = await self.remote_rpc_packager(self.find_value, args=(address, self.source_node.id, node_to_find.id))
        if result[0] and isinstance(result[1], float):
            data = self.temp_data[result[1]].result().decode()
            log.debug("get result from {}:{}".format(address[0], address[1]))
            self.temp_data.pop(result[1])
            result = (True, {'value': data})
        return self.handle_call_response(result, node_to_ask)

    async def call_ping(self, node_to_ask):
        address = (node_to_ask.ip, node_to_ask.port)
        result = await self.remote_rpc_packager(self.ping, args=(address, self.source_node.id))
        return self.handle_call_response(result, node_to_ask)

    async def call_store(self, node_to_ask, key, value):
        address = (node_to_ask.ip, node_to_ask.port)
        link_key = await self.call_send_data(address, value)
        if not link_key:
            return self.handle_call_response((False, False), node_to_ask)
        result = await self.remote_rpc_packager(self.store, args=(address, self.source_node.id, key, link_key))
        return self.handle_call_response(result, node_to_ask)
    
    async def call_send_data(self, receiver, data):
        result = await self.remote_rpc_packager(self.send_prepar, args=(receiver, len(data)))
        if result[0]:
            result = result[1]
            sender = DataChannelSender(result['port'], result['ip'])
            if not sender.send(data.encode()):
                return False
            return result['key']

    def welcome_if_new(self, node):
        """
        Given a new node, send it all the keys/values it should be storing,
        then add it to the routing table.

        @param node: A new node that just joined (or that we just found out
        about).

        Process:
        For each key in storage, get k closest nodes.  If newnode is closer
        than the furtherst in that list, and the node for this server
        is closer than the closest in that list, then store the key/value
        on the new node (per section 2.5 of the paper)
        """
        if not self.router.is_new_node(node):
            return

        log.info("never seen %s before, adding to router", node)
        for key, value in self.storage:
            keynode = Node(digest(key))
            neighbors = self.router.find_neighbors(keynode)
            if neighbors:
                last = neighbors[-1].distance_to(keynode)
                new_node_close = node.distance_to(keynode) < last
                first = neighbors[0].distance_to(keynode)
                this_closest = self.source_node.distance_to(keynode) < first
            if not neighbors or (new_node_close and this_closest):
                asyncio.ensure_future(self.call_store(node, key, value))
        self.router.add_contact(node)

    async def remote_rpc_packager(self, rpc_method, args, times=5):
        for i in range(times):
            try:
                result = await rpc_method(*args)
                if result[0]:
                    return result
            except Exception as e:
                log.error("remote process call error,")
                print(e)
            await asyncio.sleep(3)
        return [False, ]

    def handle_call_response(self, result, node):
        """
        If we get a response, add the node to the routing table.  If
        we get no response, make sure it's removed from the routing table.
        """
        if not result[0]:

            log.warning("no response from %s, removing from router", node)
            self.router.remove_contact(node)
            return result

        log.info("got successful response from {}:{}".format(node, result))
        self.welcome_if_new(node)
        return result
    
    
confirm_msg = b'/-ok!-/'

class DataChenelReceiver():
    def __init__(self, port, datasize, host="127.0.0.1") -> None:
        self.host = host
        self.port = port
        self.datasize = datasize


    async def receiver_handler(self, reader, writer):
        self.data = await asyncio.wait_for(reader.read(self.datasize), 5)
        if not self.data:
            writer.close()
            await writer.wait_closed()
            return
        writer.write(confirm_msg)
        await writer.drain()
        writer.close()
        self.close()


    def run(self):
        loop = asyncio.get_event_loop()
        server = asyncio.start_server(self.receiver_handler, host=self.host, port=self.port, loop=loop)
        self.server = loop.run_until_complete(server)
        
    def close(self):
        self.server.close()

    def result(self) -> str:
        return self.data

class DataChannelSender():
    def __init__(self, port, host="127.0.0.1") -> None:
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5)
    
    def send(self, data:bytes):
        try:
            self.sock.connect((self.host, self.port))
        except ConnectionRefusedError:
            log.info("can not connect remote tcp server..")
            return False
        self.sock.sendall(data)
        resp = self.sock.recv(len(confirm_msg))
        if resp == confirm_msg:
            return True
        return False