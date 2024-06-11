#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 8:01
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/load_toml.py
# @Software: VS Code

import tkinter as tk

from tkinter import filedialog

from main_page import MainPage


class LoadToml:
    def __init__(self, _root: tk.Tk):
        """初始化预测页面

                Args:
                    _root: `tkinter.Tk` 类型，创建所需 TK 页面
                """
        self.__root = _root
        # 设置窗口的初始化大小为 300 像素 * 200 像素
        self.__root.geometry('300x200')
        # 设置窗口的标题
        self.__root.title("Myriad v0.1.0")

        # 设置按钮和文字的布局
        self.__loadtoml_frame = tk.Frame(self.__root)
        self.__loadtoml_frame.grid(padx='20', pady='30')
        self.__label = tk.Label(self.__loadtoml_frame, text="请上传参数配置的 `toml` 文件")
        self.__label.grid(row=0, column=1, padx='20', pady='30')
        self.__btn = tk.Button(self.__loadtoml_frame, text='上传文件', command=self.__upload_file)
        self.__btn.grid(row=1, column=0, ipadx='3', ipady='3', padx='0', pady='20')
        self.__entry = tk.Entry(self.__loadtoml_frame, width=20)
        self.__entry.grid(row=1, column=1)

    def __upload_file(self):
        """让用户上传 `toml` 文件，最终获得 `toml` 文件的路径，给 `MainPage` 类读取

        Returns:

        """
        select_file = tk.filedialog.askopenfilename()
        self.__entry.insert(0, select_file)
        print(f'上传文件: {self.__entry.get()}')
        toml_path = str(self.__entry.get())
        self.__loadtoml_frame.destroy()
        MainPage(self.__root, toml_path)


if __name__ == '__main__':
    # 创建 `tkinter` 中的 `TK` 类，以实现窗口
    root = tk.Tk()
    LoadToml(_root=root)
    root.mainloop()
