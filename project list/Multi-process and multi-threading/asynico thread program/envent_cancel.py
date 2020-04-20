import asyncio, time


async def work(id, t):
    print('Working...')
    await asyncio.sleep(t)
    print('Work {} done'.format(id))

def main_1():
    start = time.time()

    loop = asyncio.get_event_loop()
    coroutines = [work(i, i) for i in range(1, 4)]            # 1
    try:
        loop.run_until_complete(asyncio.gather(*coroutines))  # 2
    except KeyboardInterrupt:
        loop.stop()    # 3
    finally:
        loop.close()   # 4

    end = time.time()
    print("time cost:   ", end - start)


def main_2():
    start = time.time()

    loop = asyncio.get_event_loop()
    coroutines = [work(i, i) for i in range(1, 4)]
    # 程序运行过程中，快捷键 Ctrl + C 会触发 KeyboardInterrupt 异常
    try:
        loop.run_until_complete(asyncio.gather(*coroutines))
    except KeyboardInterrupt:
        print()
        # 每个线程里只能有一个事件循环
        # 此方法可以获得事件循环中的所有任务的集合
        # 任务的状态有 PENDING 和 FINISHED 两种
        tasks = asyncio.Task.all_tasks()
        for i in tasks:
            print('取消任务：{}'.format(i))
            # 任务的 cancel 方法可以取消未完成的任务
            # 取消成功返回 True ，已完成的任务取消失败返回 False
            print('取消状态：{}'.format(i.cancel()))
    finally:
        loop.close()

    end = time.time()
    print("time cost:   ", end - start)


if __name__ == '__main__':
    # main_1()
    main_2()