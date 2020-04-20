import asyncio
import time

l = []
lock = asyncio.Lock()   # 协程锁


async def work(name):
    print('lalalalalalalala')     # 打印此信息是为了测试协程锁的控制范围
    # 这里加个锁，第一次调用该协程，运行到这个语句块，上锁
    # 当语句块结束后解锁，开锁前该语句块不可被运行第二次
    # 如果上锁后有其它任务调用了这个协程函数，运行到这步会被阻塞，直至解锁
    # with 是普通上下文管理器关键字，async with 是异步上下文管理器关键字
    # 能够使用 with 关键字的对象须有 __enter__ 和 __exit__ 方法
    # 能够使用 async with 关键字的对象须有 __aenter__ 和 __aexit__ 方法
    # async with 会自动运行 lock 的 __aenter__ 方法，该方法会调用 acquire 方法上锁
    # 在语句块结束时自动运行 __aexit__ 方法，该方法会调用 release 方法解锁
    # 这和 with 一样，都是简化 try ... finally 语句
    async with lock:
        print('{} start'.format(name))  # 头一次运行该协程时打印
        if 'x' in l:                    # 如果判断成功
            return name                 # 直接返回结束协程，不再向下执行
        await asyncio.sleep(0); print('----------')  # 阻塞 0 秒，切换协程
        l.append('x')
        print("l: ", l)
        print('{} end'.format(name))
        return name

async def one():
    name = await work('one')
    print('{} ok'.format(name))

async def two():
    name = await work('two')
    print('{} ok'.format(name))

def main():
    start = time.time()

    loop = asyncio.get_event_loop()
    tasks = asyncio.wait([one(), two()])
    loop.run_until_complete(tasks)

    end = time.time()
    print('耗时：{:.4f}s'.format(end - start))
    # print(l)


if __name__ == '__main__':
    main()