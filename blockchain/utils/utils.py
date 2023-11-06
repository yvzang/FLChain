import hashlib
import asyncio

def digest(string):
    if not isinstance(string, bytes):
        string = str(string).encode('utf8')
    return hashlib.sha1(string).digest()

def data_digest(string) -> int:
    return digest(string).hex()

async def gather_syncronize_task(tasks : list):
    return await asyncio.gather(*tasks)