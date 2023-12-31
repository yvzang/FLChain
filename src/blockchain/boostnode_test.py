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


def command(node : Node, main_loop):
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
            future = asyncio.run_coroutine_threadsafe(node.make_transaction(args.to, args.data), main_loop)
            trans = future.result(5)
            asyncio.run_coroutine_threadsafe(node.broadcast_transaction(trans), main_loop)
        elif args.cmd == 'get':
            args.add_argument('-h', '--hash', arg_type=str, help="The hash of data")
            node.get_value(args.hash)
        elif args.cmd == 'create_block':
            block = node.make_block(node.trans_pool)
            if block:
                asyncio.run_coroutine_threadsafe(node.broadcast_block(block), main_loop)
        elif args.cmd == 'bootstrap':
            args.add_argument('-i', '--inter', arg_type=str, help='The bootsrap ip.')
            args.add_argument('-p', '--port', arg_type=int, help='The bootstrap port.')
            asyncio.run_coroutine_threadsafe(node.node_bootstrap([(args.inter, args.port)]), main_loop)
        elif args.cmd == 'nei':
            node.print_neighbor()
        


def create_node(loop, node : Node, interface, port, boot_inter, boot_port):
    asyncio.set_event_loop(loop)

    loop.run_until_complete(node.listen(port, interface))
    if boot_inter and boot_port:
        boostaddress = (boot_inter, int(boot_port))
        loop.run_until_complete(node.node_bootstrap([boostaddress]))
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        loop.close()

def run(args):
    main_loop = asyncio.new_event_loop()
    node = Node(node_id=args.id)
    thread = Thread(target=create_node, args=[main_loop, node, args.inter, args.port, args.boot_inter, args.boot_port], daemon=True)
    thread.start()
    command(node, main_loop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inter', type=str, default="127.0.0.1", help="a ip binded to listen message")
    parser.add_argument('-p', '--port', type=int, default=8897, help="a port binded to listen message.")
    parser.add_argument('-n', '--id', type=str, default='88888888dd', help="a node id.")
    parser.add_argument('-bi', '--boot_inter', type=str, default="127.0.0.1", help="a ip address to booststrap a node.")
    parser.add_argument('-bp', '--boot_port', type=int, default=None, help="a port to booststrap a node.")
    parser.parse_args()
    args = parser.parse_args()
    run(args)
