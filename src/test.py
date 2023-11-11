import torch
from io import BytesIO


if __name__ == '__main__':
    path = r"test.pt"
    ten = {'key':torch.Tensor([1, 2, 3])}
    print(ten)
    byte_io = BytesIO()
    torch.save(ten, byte_io)
    byte_io.seek(0)
    byte_hex = byte_io.getvalue().hex()
    byte = bytes.fromhex(byte_hex)
    byte_io = BytesIO(byte)
    byte_io.seek(0)
    ten = torch.load(byte_io)
    print(ten)
    with open(path, 'r') as f:
        file_str = f.read()
        buffer = BytesIO(bytes.fromhex(file_str))
        buffer.seek(0)
    ten = torch.load(buffer)
    print(ten)
