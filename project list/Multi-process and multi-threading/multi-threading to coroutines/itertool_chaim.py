"""内置模块 itertools  chain 方法，它可以接受任意数量的可迭代对象作为参数，
返回一个包含所有参数中的元素的迭代器："""
from itertools import chain


c = chain({'one', 'two'}, list('abc'))
print(c)
for i in c:
    print(i)

print("**"*20)

n = [{'one', 'two'}, list('abc')]
c1 = chain(n)
print(c1)   # 迭代器
for j in c1:
    print(j)

print("**"*20)

def chain_(*args):
    for iter_obj in args:
        for i in iter_obj:
            yield i

c = chain_(n)
print(c)        # 生成器
for i in c:
    print(i)

print("**"*20)

def chain_2(*args):
    for iter_obj in args:
        yield from iter_obj   # 生成迭代器

c = chain_2(n)
print(c)
for i in c:
    print(i)