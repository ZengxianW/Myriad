#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 3:49
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/about_frame.py
# @Software: VS Code

import tkinter as tk


class AboutFrame(tk.Frame):
    """关于界面的包装实现

    """

    def __init__(self, _root):
        # 执行父类的构造函数，使得我们能够调用父类的属性。
        super().__init__(_root)
        tk.Label(self, text="作者：王增仙").pack()
        tk.Label(self, text="学号：20221030120").pack()
        tk.Label(self, text="学院：经济学院").pack()
        tk.Label(self, text="专业：金融学").pack()
        tk.Label(self,
                 text="关于作品：Myriad，你的金融助手，能够帮助你预测明天的股价，也能使用 ChatGPT 作为你的金融导师").pack()


if __name__ == '__main__':
    pass
