# coding: utf-8

import torch
import torch.nn as nn

# ----------------------------------- L1 Loss

# 生成网络输出 以及 目标输出
output = torch.ones(2, 2, requires_grad=True)*0.5
target = torch.ones(2, 2)

# 设置三种不同参数的L1Loss
reduce_False = nn.L1Loss(reduction='none')
size_average_True = nn.L1Loss(reduction='mean')
size_average_False = nn.L1Loss(reduction='sum')

o_0 = reduce_False(output, target)
o_1 = size_average_True(output, target)
o_2 = size_average_False(output, target)

print('\nreduce="none", 输出同维度的loss:\n{}\n'.format(o_0))
print('reduction="mean"，\t求平均:\t{}'.format(o_1))
print('reduction="sum"，\t求和:\t{}'.format(o_2))
