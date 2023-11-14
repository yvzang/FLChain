import time
from itertools import takewhile
import operator
import binascii
import os
from os import path
from collections import OrderedDict
from abc import abstractmethod, ABC


class IStorage(ABC):
    """
    Local storage for this node.
    IStorage implementations of get must return the same type as put in by set
    """

    @abstractmethod
    def __setitem__(self, key, value):
        """
        Set a key to the given value.
        """

    @abstractmethod
    def __getitem__(self, key):
        """
        Get the given key.  If item doesn't exist, raises C{KeyError}
        """

    @abstractmethod
    def get(self, key, default=None):
        """
        Get given key.  If not found, return default.
        """

    @abstractmethod
    def iter_older_than(self, seconds_old):
        """
        Return the an iterator over (key, value) tuples for items older
        than the given secondsOld.
        """

    @abstractmethod
    def __iter__(self):
        """
        Get the iterator for this storage, should yield tuple of (key, value)
        """


class ForgetfulStorage(IStorage):
    def __init__(self, storage_path, ttl=604800):
        """
        By default, max age is a week.
        """
        self.data = OrderedDict()
        self.ttl = ttl
        self.storage_path = storage_path
        self.__load_items_from_path__()

    def __load_items_from_path__(self):
        if not path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        else:
            self.clear_storage()

    def __write_to__(self, full_name, value):
        with open(full_name, mode='wb') as f:
            return f.write(value.encode())

    def __read_from__(self, full_name):
        with open(full_name, mode='rb') as f:
            return f.read().decode()

    def __setitem__(self, key, value):
        if key in self.data:
            self.remove(key)
        self.data[key] = time.monotonic()
        self.__write_to__(self.__full_name__(self.__concat_filename__(self.data[key], key)), value)
        self.cull()

    def cull(self):
        for _, _ in self.iter_older_than(self.ttl):
            item = self.data.popitem(last=False)
            self.remove(item[0])

    def get(self, key, default=None):
        self.cull()
        if key in self.data:
            return self[key]
        return default
    
    def remove(self, key):
        if key in self.data:
            os.remove(self.__full_name__(self.__concat_filename__(self.data[key], key)))
            self.data.pop(key)

    def clear_storage(self):
        file_list = os.listdir(self.storage_path)
        for filename in file_list:
            os.remove(self.__full_name__(filename))

    def __getitem__(self, key):
        self.cull()
        return self.__read_from__(self.__full_name__(self.__concat_filename__(self.data[key], key)))

    def __repr__(self):
        self.cull()
        return repr(self.data)

    def iter_older_than(self, seconds_old):
        min_birthday = time.monotonic() - seconds_old
        zipped = self.data.items()
        matches = takewhile(lambda r: min_birthday >= r[1], zipped)
        return list(matches)

    def _triple_iter(self):
        ikeys = self.data.keys()
        ibirthday = map(operator.itemgetter(0), self.data.values())
        ivalues = map(operator.itemgetter(1), self.data.values())
        return zip(ikeys, ibirthday, ivalues)

    def __iter__(self):
        for key in self.data.keys():
            yield (key, self[key])

    def __concat_filename__(self, monotonic:float, key:bytes) -> str:
        return str(monotonic)+'-'+ key.hex()
    
    def __split_filaname__(self, filename:str) -> tuple:
        item = filename.split('-')
        return (float(item[0]), bytes.fromhex(item[1]))
    
    def __full_name__(self, filename):
        return self.storage_path+'\\'+filename
        