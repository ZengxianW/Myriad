#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/9 2:23
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/myriad/main.py
# @Software: VSCode

import tkinter as tk

from gui import LoadToml


def main() -> int:
    # print("hello")
    # 创建 `tkinter` 中的 `TK` 类，以实现窗口
    root = tk.Tk()
    LoadToml(_root=root)
    root.mainloop()


if __name__ == '__main__':
    main()
