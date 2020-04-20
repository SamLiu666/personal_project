import asyncio
import time


def hello(name):          # 普通函数
    print('[hello] Hello, {}'.format(name))


async def work(t, name):  # 协程函数
    print('[work ] start', name)
    await asyncio.sleep(t)
    # time.sleep(t)
    print('[work ] {} after {}s stop'.format(name, t))

def main():
    start = time.time()

    loop = asyncio.get_event_loop()
    # 向事件循环中添加任务
    asyncio.ensure_future(work(1, 'A'))     # 第 1 个执行
    # call_soon 将普通函数当作 task 加入到事件循环并排定执行顺序
    # 该方法的第一个参数为普通函数名字，普通函数的参数写在后面
    loop.call_soon(hello, 'Tom')            # 第 2 个执行
    # 向事件循环中添加任务
    loop.create_task(work(2, 'B'))          # 第 3 个执行
    # 阻塞启动事件循环，顺便再添加一个任务
    loop.run_until_complete(work(3, 'C'))   # 第 4 个执行

    end = time.time()
    print('耗时：{:.4f}s'.format(end - start))


if __name__ == '__main__':
    main()