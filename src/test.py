import torch
from io import BytesIO


if __name__ == '__main__':
    ten = {'key':torch.Tensor([1, 2, 3])}
    print(ten)
    byte_io = BytesIO()
    torch.save(ten, 'test.pt')
    torch.save(ten, byte_io)
    print(byte_io.getvalue())
    with open('test.pt', 'rb') as f:
        buffer = BytesIO(f.read())
        print(buffer.getvalue())
    print(byte_io == buffer)
    byte_io.seek(0)
    ten = torch.load(byte_io)
    print(ten)
