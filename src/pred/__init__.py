#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 6:10
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/pred/__init__.py.py
# @Software: VS Code

__all__ = []

__description__= ("首先使用 `efinance` 与 `baostock` 两个模块获取股票信息，然后使用写好的 `lstm_model` 模块构建深度学习模型 LSTM"
                  " 进行股票数据的预测。最后实现了不同股票预测模型的多进程以加快股票的预测速度。")

__author__ = "Zengxian Wang <zengxian822.wang@gmail.com>"
__version__ = "0.1"
__date__ = "10th June 2024"