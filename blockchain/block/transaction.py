from utils.utils import *
from copy import deepcopy
import time
import json


class Transaction:
    def __init__(self, trans_from : str=None, to : str=None, data=None, stamp=None, hash=None) -> None:
        self.trans_from = trans_from
        self.to = to
        self.data = data
        self.stamp = stamp
        self.hash = hash


    @staticmethod
    def Transe_Create(trans_from : str, to : str, data : str):
        r"""creates a new transaction. The suplement message such as stamp and hash will be automaticly 
        maked. 
        Notes:
            The data must be str type and return None if not be succesfully created.
        """
        if not isinstance(data, str): None
        trans = Transaction(trans_from, to, data, time.time())
        trans.hash = trans.__digest__()
        return trans

    @staticmethod
    def Trans_Decode(string : str):
        trans_dict = json.loads(string)
        new_trans = Transaction()
        for attr, value in trans_dict.items():
            if not hasattr(new_trans, attr): 
                return None
            setattr(new_trans, attr, value)
        return new_trans
    

    def meta_data(self) -> dict:
        return {'from': self.trans_from, 'to': self.to, 'data': self.data}
    
    def __str__(self) -> str:
        return json.dumps(self.__dict__).replace('\'', '\"')
    
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Transaction):
            return self.hash == __value.hash
        return False
    
    
    def __digest__(self) -> str:
        return digest(digest(self.trans_from) + digest(self.to) + digest(self.data) + digest(self.stamp)).hex()
