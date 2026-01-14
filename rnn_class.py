import math
import torch
from torch import nn
from torch.nn import  functional as F
from d2l import torch as d2l


batch_size,num_steps=32,35#每次的批大小，时间维度大小
train_iter,vocab=d2l.load_data_time_machine(batch_size,num_steps)#调用 d2l 的工具函数加载《时光机器》这本电子书的文本

#F.one_hot(torch.tensor([0,2]),len(vocab))

#初始化模型参数
def get_params(vocab_size,num_hiddens,device):
    num_inputs=num_outputs=vocab_size

    #定义内部函数normal
    def normal(shape):
        return torch.randn(shape,device=device)*0.01

    #隐藏层参数
    W_xh=normal((num_hiddens,num_hiddens))
    W_hh=normal((num_hiddens,num_hiddens))
    b_h=torch.zeros(num_hiddens,device=device)

    #输出层参数
    W_hq=normal((num_hiddens,num_outputs))
    b_q=torch.zeros(num_outputs,device=device)

    #附加梯度
    params=[W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.requires_grad_(True)
    return params

#定义初始化函数
def init_rnn_state(batch_size,num_hiddens,device):
    return torch.zeros(batch_size,num_hiddens,device=device)

#一个时间步内，rnn是如何工作的
def rnn(inputs,state,params):
     

