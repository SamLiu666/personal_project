import functools
import time, asyncio


def one():
    start = time.time()

    @asyncio.coroutine  # 装饰器创建协程参数
    def do():           # 协程参数名称
        print("Start:   ")
        time.sleep(0.1)     # 模拟I/O操作
        print("This is a coroutine")

    loop = asyncio.get_event_loop()     # 创建事件循环
    coroutine = do()                    # 调用获取协程对象
    loop.run_until_complete(coroutine)  #将协程写入事件循环控制

    end = time.time()
    print("time costs:  ", end-start)

def one_2():
    start = time.time()

    @asyncio.coroutine  # 装饰器创建协程参数
    def do():           # 协程参数名称
        print("Start:   ")
        time.sleep(0.1)     # 模拟I/O操作
        print("This is a coroutine")

    loop = asyncio.get_event_loop()     # 创建事件循环

    coroutine = do()        # 调用获取协程对象
    task = loop.create_task(coroutine)      # 创建任务
    print('task 是不是 asyncio.Task 的实例？', isinstance(task, asyncio.Task))
    print('Task state:', task._state)

    loop.run_until_complete(task)  #将协程写入事件循环控制

    end = time.time()
    print("time costs:  ", end-start)


def three():
    start = time.time()
    # @asyncio.coroutine
    async def corowork():      # 使用async 创建协程参数
        print('[corowork] Start coroutine')
        time.sleep(0.1)
        print('[corowork] This is a coroutine')

    def callback(name, task):  # 回调函数，协程终止后需要顺便运行的代码写入这里，回调函数的参数有要求，最后一个位置参数须为 task 对象
        print('[callback] Hello {}'.format(name))
        print('[callback] coroutine state: {}'.format(task._state))

    loop = asyncio.get_event_loop()
    coroutine = corowork()
    task = loop.create_task(coroutine)
    task.add_done_callback(functools.partial(callback, 'world'))  # 3
    loop.run_until_complete(task)

    end = time.time()
    print('运行耗时：{:.4f}'.format(end - start))

def four():
    """往往有多个协程创建多个任务对象，同时在一个 loop 里运行。
    为了把多个协程交给 loop，需要借助 asyncio.gather 方法。
    任务的 result 方法可以获得对应的协程函数的 return 值"""
    print("asyncio.sleep 阻止当前线程，还可执行其他")
    start = time.time()

    # @asyncio.coroutine
    async def corowork(name, t):      # 使用async 创建协程参数
        print('[corowork] Start coroutine', name)

        # asyncio.sleep 与 time.sleep 是不同的，前者阻塞当前协程，即 corowork 函数的运行，而 time.sleep 会阻塞整个线程，所以这里必须用前者，阻塞当前协程，CPU 可以在线程内的其它协程中执行
        await asyncio.sleep(t)
        # time.sleep(t)   # 需等整个线程完成

        print('[corowork] Stop coroutine', name)
        return 'Coroutine {} OK'.format(name)

    loop = asyncio.get_event_loop()
    coroutine1 = corowork("ONE", 1)
    coroutine2 = corowork("TWO", 3)
    task1 = loop.create_task(coroutine1)
    task2 = loop.create_task(coroutine2)

    gather = asyncio.gather(task1, task2)   # 收集任务

    loop.run_until_complete(gather)
    print('[task1] ', task1.result())
    print('[task2] ', task2.result())

    end = time.time()
    print('运行耗时：{:.4f}'.format(end - start))


def four_2():
    """往往有多个协程创建多个任务对象，同时在一个 loop 里运行。
    为了把多个协程交给 loop，需要借助 asyncio.gather 方法。
    任务的 result 方法可以获得对应的协程函数的 return 值"""
    print("time.sleep 阻止整个线程，需执行完")
    start = time.time()

    # @asyncio.coroutine
    async def corowork(name, t):      # 使用async 创建协程参数
        print('[corowork] Start coroutine', name)

        # asyncio.sleep 与 time.sleep 是不同的，前者阻塞当前协程，即 corowork 函数的运行，而 time.sleep 会阻塞整个线程，所以这里必须用前者，阻塞当前协程，CPU 可以在线程内的其它协程中执行
        # await asyncio.sleep(t)
        time.sleep(t)   # 需等整个线程完成

        print('[corowork] Stop coroutine', name)
        return 'Coroutine {} OK'.format(name)

    loop = asyncio.get_event_loop()
    coroutine1 = corowork("ONE", 1)
    coroutine2 = corowork("TWO", 3)
    task1 = loop.create_task(coroutine1)
    task2 = loop.create_task(coroutine2)

    gather = asyncio.gather(task1, task2)   # 收集任务

    loop.run_until_complete(gather)
    print('[task1] ', task1.result())
    print('[task2] ', task2.result())

    end = time.time()
    print('运行耗时：{:.4f}'.format(end - start))

if __name__ == '__main__':

    one()
    print("**"*20)

    one_2()
    print("**"*20)

    three()
    print("**"*20)

    four()
    print("**"*20)

    four_2()