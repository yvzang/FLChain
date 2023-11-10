from blockchain.kademlia.protocol import *
import asyncio

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    re = DataChenelReceiver(8888, 17)
    re.run()
    print("...")
    loop.run_forever()
