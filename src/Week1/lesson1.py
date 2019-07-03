#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson1.py
@time: 2019/6/6 10:07
@desc: 验证安装的环境
"""
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print('hello, world.')
print(torch.cuda.current_device())