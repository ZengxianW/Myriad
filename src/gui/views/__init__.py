#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 3:56
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/views/__init__.py
# @Software: VS Code

from gui.views.about_frame import AboutFrame
from gui.views.chat_frame import ChatFrame
from gui.views.pred_frame import PredFrame
from gui.views.search_frame import SearchFrame

__all__ = ["AboutFrame", "PredFrame", "ChatFrame", "SearchFrame"]

__description__ = "`tkinter` 中的一些 `Frame` 的包装实现。"

__author__ = "Zengxian Wang <zengxian822.wang@gmail.com>"
__version__ = "0.1"
__date__ = "11st June 2024"
