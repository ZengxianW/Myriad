#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/8 22:37
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/lstm_model/__init__.py
# @Software: VSCode

# 初始化模块，定义导入的类，并设置模块的元信息，使其他模块可以方便地使用这些类。

from lstm_model.lstm_module import LstmModule
from lstm_model.pl_lstm import LitAutoLstm

# 定义了模块的公开接口，当使用 from ... import * 时，这些名称会被导入。
__all__ = ["LstmModule", "LitAutoLstm"]

__description__= "使用 `torch` 和 `pytorch lightning` 两个库实现的深度学习模型 LSTM 的构建以及训练过程的创建。"

__author__ = "Zengxian Wang <zengxian822.wang@gmail.com>"
__version__ = "0.1"
__date__ = "9th June 2024"
