import torch
from io import BytesIO

def read_binary_from_file(file_name):
    with open(file_name, 'rb') as f:
        return f.read()
    

def ten2str(ten_like):
    byte_io = BytesIO()
    torch.save(ten_like, byte_io)
    del ten_like
    return byte_io.getvalue().hex()


def str2ten(ten_str):
    byte_io = BytesIO(bytes.fromhex(ten_str)); byte_io.seek(0)
    return torch.load(byte_io)