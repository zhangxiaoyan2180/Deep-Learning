import math
import torch
from torch import nn
from torch.nn import  functional as F
from d2l import torch as d2l
#
# 1. 运行函数
train_iter, vocab = d2l.load_
#2. 强行让“生成器”吐出一组数据
# next(iter(...)) 就像在售货机上按了一下“出货”键
X, Y = next(iter(train_iter))

# 3. 打印出来看
print("这是输入矩阵 X 的形状:", X.shape) # 输出 [32, 35]
print("这是第 1 条数据的数字序列:\n", X[0])
print("这是对应的预测目标 Y 的数字序列:\n", Y[0])

# 4. 用密码本翻译回人话
print("翻译 X[0]:", ''.join([vocab.idx_to_token[int(i)] for i in X[0]]))
print("翻译 Y[0]:", ''.join([vocab.idx_to_token[int(i)] for i in Y[0]]))


# import d2l.torch as d2l  # 注意这里是 d2l.torch
#
# # 再次打印看看，这次你会看到满屏的函数名
# print(dir(d2l))