import asyncio

async def coroutine_example():
    await asyncio.sleep(1)
    print('zhihu ID: Zarten')

async def wait():
    await asyncio.sleep(3)

coro = coroutine_example()

loop = asyncio.get_event_loop()
task = loop.create_task(coro)
print('运行情况：', task)

asyncio.run(wait())
print('再看下运行情况：', task)
loop.close()