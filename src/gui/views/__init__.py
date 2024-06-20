#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 3:56
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/views/__init__.py
# @Software: VS Code


# 初始化模块，定义导入的类，并设置模块的元信息，使其他模块可以方便地使用这些类


# 从 about_frame 模块导入 AboutFrame 类
from gui.views.about_frame import AboutFrame
# 从 chat_frame 模块导入 ChatFrame 类
from gui.views.chat_frame import ChatFrame
# 从 pred_frame 模块导入 PredFrame 类
from gui.views.pred_frame import PredFrame
# 从 search_frame 模块导入 SearchFrame 类
from gui.views.search_frame import SearchFrame

# 定义了模块的公开接口，当使用 from ... import * 时，这些名称会被导入
__all__ = ["AboutFrame", "PredFrame", "ChatFrame", "SearchFrame"]

__description__ = "`tkinter` 中的一些 `Frame` 的包装实现。"

__author__ = "Zengxian Wang <zengxian822.wang@gmail.com>"
__version__ = "0.1"
__date__ = "11st June 2024"
