# curses 用来在终端上显示图形界面
# import curses
# random 模块用来生成随机数
from random import randrange, choice
# collections 提供了一个字典的子类 defaultdict。可以指定 key 值不存在时，value 的默认值。
from collections import defaultdict


# ord() 函数以一个字符作为参数，返回参数对应的 ASCII 数值，便于和后面捕捉的键位关联
letter_codes = [ord(ch) for ch in 'WASDRQwasdrq']
actions = ['Up', 'Left', 'Down', 'Right', 'Restart', 'Exit']
actions_dict = dict(zip(letter_codes, actions * 2))
print(actions_dict)