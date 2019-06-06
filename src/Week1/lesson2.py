#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: lesson2.py
@time: 2019/6/6 10:14
@desc: 自动求导机制
"""

# 从后向中排除子图
import torch
import torchvision

regular_input = torch.randn(1, 3, 224, 224)  # 默认是True
volatile_input = torch.randn(1, 3, 224, 224)
model = torchvision.models.resnet18(pretrained=True)
model(regular_input).requires_grad
with torch.no_grad():
    model(volatile_input).requires_grad
