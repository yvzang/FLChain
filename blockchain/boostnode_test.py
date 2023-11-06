from node import Node
from block.transaction import Transaction
from block.transaction_pool import Transaction_pool
from block.block import Block
import logging
import base64
import argparse
from parse import ArgParse
from threading import Thread
import asyncio

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log = logging.getLogger('Blockchain')
log.addHandler(handler)
log.setLevel(logging.DEBUG)


async def command(node : Node):
    while True:
        cmd = input("Input your command:\n")
        args = ArgParse(cmd)
        args.add_argument('-c', '--cmd', arg_type=str, help='The operate command.')
        if args.cmd == 'headers':
            node.print_block_header()
        elif args.cmd == 'check_block':
            args.add_argument('-h', '--hash', arg_type=str, help='The block hash want to check.')
            node.print_block(args.hash)
        elif args.cmd == 'tx_pool':
            node.print_transaction_pool()
        elif args.cmd == 'create_tx':
            args.add_argument('-t', '--to', arg_type=str, help='The receiver of the transaction.')
            args.add_argument('-d', '--data', arg_type=str, help="data")
            trans = node.make_transaction(args.to, args.data)
            asyncio.run(node.broadcast_transaction(trans))
        elif args.cmd == 'get':
            args.add_argument('-h', '--hash', arg_type=str, help="The hash of data")
            node.get_value(args.hash)
        elif args.cmd == 'create_block':
            block = node.make_block(node.trans_pool)
            asyncio.run(node.broadcast_block(block))
        elif args.cmd == 'bootstrap':
            args.add_argument('-i', '--inter', arg_type=str, help='The bootsrap ip.')
            args.add_argument('-p', '--port', arg_type=int, help='The bootstrap port.')
            asyncio.run(node.node_bootstrap([(args.inter, args.port)]))
        


def create_node(loop, node : Node, interface, port, boot_inter, boot_port):
    loop.set_debug(True)

    loop.run_until_complete(node.listen(port, interface))
    if boot_inter and boot_port:
        boostaddress = (boot_inter, int(boot_port))
        print(boostaddress)
        loop.run_until_complete(node.node_bootstrap([boostaddress]))
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        loop.close()

def run(args):
    node = Node(node_id=args.id)
    loop = asyncio.get_event_loop()
    thread = Thread(target=create_node, args=[loop, node, args.inter, args.port, args.boot_inter, args.boot_port], daemon=True)
    thread.start()
    asyncio.run(command(node))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inter', type=str, default="127.0.0.1", help="a ip binded to listen message")
    parser.add_argument('-p', '--port', type=int, default=8888, help="a port binded to listen message.")
    parser.add_argument('-n', '--id', type=str, default='88888888', help="a node id.")
    parser.add_argument('-bi', '--boot_inter', type=str, default="127.0.0.1", help="a ip address to booststrap a node.")
    parser.add_argument('-bp', '--boot_port', type=int, default=None, help="a port to booststrap a node.")
    parser.parse_args()
    args = parser.parse_args()
    run(args)
