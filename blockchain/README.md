# blockchain
This is a distributed data storage system with fully decentralized blockchain.  
This library is a asynchronous Python implementation with [asyncio](https://docs.python.org/3/library/asyncio.html) library based [Kademlia](https://github.com/bmuller/kademlia)  
# Installation
```python
git clone https://github.com/yvzang/blockchain.git
```
#  Usage  
##  Bootstrap  
You can bootstrap a initial node and use command fallowing:
```python
>>>python boostnode_test.py -i 127.0.0.1 -p 8888 -n 8888888
2023-11-05 21:31:39,025 - Blockchain - INFO - Node 4051049678932293688 listening on 127.0.0.1:8888
2023-11-05 21:31:39,041 - Blockchain - INFO - There is no beighbors to syncronize.
Input your command:  
```
where it will bootstrap a node server named 88888888 with address "127.0.0.1":8888.  
You can also start a new node accoding the initial node address:
```python
>>>python boostnode_test.py -i 127.0.0.1 -p 8890 -n 88888890 -bi 127.0.0.1 -bp 8888
2023-11-05 21:36:20,731 - Blockchain - INFO - Node 4051049678932293936 listening on 127.0.0.1:8890
2023-11-05 21:36:20,731 - Blockchain - INFO - There is no neighbors to syncronize.
2023-11-05 21:36:20,731 - Blockchain - INFO - Send a request for state of chain from 127.0.0.1:8888
2023-11-05 21:36:20,731 - Blockchain - INFO - Successfully to init blockchian status.
Input your command:
```
It is notable that you have to provide a valid bootstrap node if a node want to connect to network and contact with other neighbors.
## making transaction  
You can input commands to create a new transaction after initialization as following:
```python
Input your command:
>>>-c create_tx -t 88888888 -d this_is_a_message.
2023-11-05 21:45:11,911 - Blockchain - INFO - Send a transaction message to 127.0.0.1:8888
Input your command:
```
where the receiver of the tx be indicated by -t and -d illustrated as storaged message.
The command as following illustrates how to check the transaction pool:
```python
Input your command:
>>>-c tx_pool
[{'trans_from': '88888890', 'to': '88888888', 'data': 'c2543fff3bfa6f144c2f06a7de6cd10c0b650cae', 'stamp': 1699191911.8724878, 'hash': 
'd5e91b26407df42df8ce70d8ae87a13f66b81024'}]
```
It is notable that the field of "data" is not a real data storaged in blockchain, and it just a hash of data in reality.
One must provide a hash value above to check the real data:
```python
>>>-c get -h c2543fff3bfa6f144c2f06a7de6cd10c0b650cae
this_is_a_message
```
## making block
We not provid a valid consensus algorithm to create a new block for its expandability. So one can create new blocks for any node or anytime as following:
```python
>>>-c create_block
```
and print all block headers accoring to:
```python
>>>-c headers
{'trans_num': 1, 'id': 0, 'stamp': 1699193428.7065442, 'creater': '88888890', 'pre_hash': None, 'hash': '467a361f556a64be60a7b3d261fb143fda1189e5', 'next_hash': None}
```
You can check the block details by providing hash of block:
```python
>>>-c check_block -h 467a361f556a64be60a7b3d261fb143fda1189e5
header:
{'trans_num': 1, 'id': 0, 'stamp': 1699193428.7065442, 'creater': '88888890', 'pre_hash': None, 'hash': '467a361f556a64be60a7b3d261fb143fda1189e5', 'next_hash': None}
body:
[{'trans_from': '88888890', 'to': '88888888', 'data': 'c2543fff3bfa6f144c2f06a7de6cd10c0b650cae', 'stamp': 1699191911.8724878, 'hash': 
'd5e91b26407df42df8ce70d8ae87a13f66b81024'}]
```
