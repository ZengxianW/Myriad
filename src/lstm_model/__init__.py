#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/8 22:37
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/lstm_model/__init__.py
# @Software: VSCode

from lstm_model.lstm_module import LstmModule
from lstm_model.pl_lstm import LitAutoLstm

__all__ = ["LstmModule", "LitAutoLstm"]

__description__= "使用 `torch` 和 `pytorch lightning` 两个库实现的深度学习模型 LSTM 的构建以及训练过程的创建。"

__author__ = "Zengxian Wang <zengxian822.wang@gmail.com>"
__version__ = "0.1"
__date__ = "9th June 2024"
