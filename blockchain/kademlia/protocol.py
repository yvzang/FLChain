import random
import asyncio
import logging

from rpcudp.protocol import RPCProtocol

from utils.utils import *
from kademlia.node import Node
from kademlia.routing import RoutingTable
from kademlia.utils import digest

log = logging.getLogger(__name__)  # pylint: disable=invalid-name


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

    
    def rpc_get_data(self, sender, key, value, hash):
        if not data_digest(value) == hash: return False
        if key not in self.temp_data.keys():
            self.temp_data[key] = value
        else:
            self.temp_data[key] = self.temp_data[key] + value
        return True
    
    def rpc_push_data(self, sender, key, package_size):
        if key not in self.temp_data.keys():
            log.error("There is not pushed data key: {}".format(key))
            return False
        value = self.temp_data[key]
        if len(value) > package_size:
            package = value[:package_size]
            self.temp_data[key] = value[package_size:]
            return {'data': package, 'finish': False, 'hash': data_digest(package)}
        else:
            package = self.temp_data[key]
            self.temp_data.pop(key)
            return {'data': package, 'finish': True, 'hash': data_digest(package)}

    def rpc_stun(self, sender):  # pylint: disable=no-self-use
        return sender

    def rpc_ping(self, sender, nodeid):
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        return self.source_node.id

    def rpc_store(self, sender, nodeid, key, link_key):
        source = Node(nodeid, sender[0], sender[1])
        self.welcome_if_new(source)
        log.debug("got a store request from %s, storing '%s'",
                  sender, key.hex())
        self.storage[key] = self.temp_data[link_key]
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
        link_key = random.random()
        self.temp_data[link_key] = value
        return link_key

    async def call_find_node(self, node_to_ask, node_to_find):
        address = (node_to_ask.ip, node_to_ask.port)
        result = await self.find_node(address, self.source_node.id,
                                      node_to_find.id)
        return self.handle_call_response(result, node_to_ask)

    async def call_find_value(self, node_to_ask, node_to_find):
        address = (node_to_ask.ip, node_to_ask.port)
        result = await self.find_value(address, self.source_node.id,
                                       node_to_find.id)
        if result[0] and isinstance(result[1], float):
            data = await self.handle_data_receiver(address, result[1])
            result = (True, {'value': data})
        return self.handle_call_response(result, node_to_ask)

    async def call_ping(self, node_to_ask):
        address = (node_to_ask.ip, node_to_ask.port)
        result = await self.ping(address, self.source_node.id)
        return self.handle_call_response(result, node_to_ask)

    async def call_store(self, node_to_ask, key, value):
        address = (node_to_ask.ip, node_to_ask.port)
        link_key = random.random()
        await self.handle_data_sender(address, link_key, value)
        result = await self.store(address, self.source_node.id, key, link_key)
        return self.handle_call_response(result, node_to_ask)

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

    def handle_call_response(self, result, node):
        """
        If we get a response, add the node to the routing table.  If
        we get no response, make sure it's removed from the routing table.
        """
        if not result[0]:
            log.warning("no response from %s, removing from router", node)
            self.router.remove_contact(node)
            return result

        log.info("got successful response from %s", node)
        self.welcome_if_new(node)
        return result
    
    async def handle_data_sender(self, receiver, link_key, value, package_size=7*1024):
        while True:
            if len(value) > package_size:
                package = value[:package_size]
            else:
                package = value
            resp = await self.get_data(receiver, link_key, package, data_digest(package))
            if not resp[0] or not resp[1]: continue
            if len(value) > package_size: value = value[package_size:]
            else: break
            

    async def handle_data_receiver(self, receiver, link_key, package_size=7*1024):
        result = ""
        while True:
            data = await self.push_data(receiver, link_key, package_size)
            if not data[0]: continue
            data = data[1]
            if not data:
                raise KeyError("Invalid key in remote server.")
            if not data['hash'] == data_digest(data['data']): continue
            result += data['data']
            if data['finish']: break
        return result