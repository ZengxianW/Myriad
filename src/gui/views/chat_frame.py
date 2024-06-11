#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 4:03
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/chat_frame.py
# @Software: VS Code

import tkinter as tk


class ChatFrame(tk.Frame):
    """聊天界面的包装实现

    """

    def __init__(self, _root):
        # 执行父类的构造函数，使得我们能够调用父类的属性。
        super().__init__(_root)
