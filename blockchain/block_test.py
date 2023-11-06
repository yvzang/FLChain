from block.transaction import *
from block.transaction_pool import *
from block.block import *
from block.blockchain import *
from copy import deepcopy



if __name__ == '__main__':
    t1 = Transaction.Transe_Create(123, 456, 7)
    t2 = deepcopy(t1)
    pool = Transaction_pool([t1, t2])
    print(pool)
    print(pool.ishas(t1))