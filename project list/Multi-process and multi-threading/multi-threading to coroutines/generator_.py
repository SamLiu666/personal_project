import inspect
from functools import wraps


def fibo():
    def fibonacci(n):
        a,b = 0,1
        while b<n:
            a,b = b, a+b
            yield a # 生成器

    f = fibonacci(100)
    print(f)
    for i in f:
        print(i)


def genator_cor():
    def generator():
        i = "activation"
        while True:
            try:
                value = yield i
            except ValueError:
                print("OVER")
            i = value

    g = generator()
    print(g)
    gen_create = inspect.getgeneratorstate(g) # 创建生成器，协程
    gen_running = next(g) # 激活生成器
    gen_suspended = inspect.getgeneratorstate(g)    # yield 处停止
    gen_send = g.send("Hello world!")

    print(gen_create, gen_running, gen_suspended, gen_send, sep="\n")

    g.throw(ValueError)     # 抛出异常
    g.close()  # 6 终止
    gen_close = inspect.getgeneratorstate(g)
    print(gen_close)

def coroutine(func):
    @wraps(func)    # wraps 装饰器保证func 函数的签名不被修改
    def wrapper(*args, **kw):
        g = func(*args, **kw)   # 可变参数，字典参数"https://n3xtchen.github.io/n3xtchen/python/2014/08/08/python-args-and-kwargs"
        next(g)
        return g
    return wrapper

@coroutine  # 装饰器重新创建协程，直接进入挂起状态
def generator():
    i = "activation"
    while True:
        try:
            value = yield i
        except ValueError:
            print("OVER")
        i = value

g = generator()

g_suspended = inspect.getgeneratorstate(g)
# print(g_suspended)


@coroutine
def genera():
    l = []
    while True:
        value = yield   # yield 表达式不弹出值，仅作暂停之用
        if value == "CLOSE":    # 如果 send 方法的参数为 CLOSE ，break 终止 while 循环，停止生成器，抛出 StopIteration 异常
            break
        l.append(value)
    return l

g = genera()
g1 = g.send("hello")
g2 = g.send("world")
print(g1,'\n', g2)
# g.send('CLOSE')       # 一旦终止则整个程序终止

print("*"*20)

g0 = genera()
for i in ('hello', 'world', 'CLOSE'):
    try:
        g0.send(i)
    except StopIteration as e:
        value = e.value
        print("END")

print("分割线：", value)