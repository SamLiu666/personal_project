import time
import asyncio
import functools


def loop_stop(loop, future):    # 函数的最后一个参数须为 future / task
    loop.stop()                 # 停止事件循环，stop 后仍可重新运行


async def work(t):              # 协程函数
    print('start')
    await asyncio.sleep(t)      # 模拟 IO 操作
    print('after {}s stop'.format(t))


def main():
    loop = asyncio.get_event_loop()
    # 创建任务收集器，参数为任意数量的协程，任务收集器本身也是 task / future 对象
    tasks = asyncio.gather(work(1), work(2))
    # 任务收集器的 add_done_callback 方法添加回调函数
    # 当所有任务完成后，自动运行此回调函数
    # 注意 add_done_callback 方法的参数是回调函数
    # 这里使用 functools.partial 方法创建偏函数以便将 loop 作为参数加入
    tasks.add_done_callback(functools.partial(loop_stop, loop))
    loop.run_forever()  # 无限运行事件循环，直至 loop.stop 停止
    loop.close()        # 关闭事件循环


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('耗时：{:.4f}s'.format(end - start))