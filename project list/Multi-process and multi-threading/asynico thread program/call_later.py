import asyncio, time
import functools

"""call_soon 立刻执行，call_later 延时执行，call_at 在某时刻执行"""

def hello(name):            # 普通函数
    print('[hello]  Hello, {}'.format(name))


async def work(t, name):    # 协程函数
    print('[work{}]  start'.format(name))
    await asyncio.sleep(t)
    # time.sleep(t)
    print('[work{}]  stop'.format(name))


def main():
    start = time.time()

    loop = asyncio.get_event_loop()
    asyncio.ensure_future(work(1, 'A'))         # 任务 1
    loop.call_later(1.2, hello, 'Tom')          # 任务 2，从事件循环启动时，计时1.2秒后执行
    loop.call_soon(hello, 'Kitty')              # 任务 3
    task4 = loop.create_task(work(2, 'B'))      # 任务 4
    loop.call_later(1, hello, 'Jerry')          # 任务 5
    loop.run_until_complete(task4)

    end = time.time()
    print('耗时：{:.4f}s'.format(end - start))


if __name__ == '__main__':
    main()

# [workA]  start
# [hello]  Hello, Kitty
# [workB]  start
# [hello]  Hello, Jerry
# [workA]  stop
# [hello]  Hello, Tom
# [workB]  stop
# 耗时：2.0003s
