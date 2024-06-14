#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 2:36
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/main_page.py
# @Software: VS Code

import tomllib

import tkinter as tk

from gui.views import AboutFrame, PredFrame, ChatFrame, SearchFrame


class MainPage:
    def __init__(self, _root: tk.Tk, _toml_path: str = "./config.toml"):
        """初始化预测页面

        Args:
            _root: `tkinter.Tk` 类型，创建所需 TK 页面
            _toml_path: 作为参数的 `toml` 文件位置
        """
        self.__root = _root
        self.__toml_path = _toml_path
        # self.__cfg = self.__load_toml()
        # 设置窗口的初始化大小为 800 像素 * 600 像素
        self.__root.geometry('800x600')
        # 设置窗口的标题
        self.__root.title("Myriad v0.1.0")

        self.__search_frame = SearchFrame(self.__root)
        self.__pred_frame = PredFrame(self.__root, self.__toml_path)
        self.__chat_frame = ChatFrame(self.__root, self.__toml_path)
        self.__about_frame = AboutFrame(self.__root)
        self.__create_menubar()

    # def __load_toml(self):
    #     # 加载配置文件，深度学习模型 LSTM 的一些超参数
    #     with open(self.__toml_path, "rb") as f:
    #         cfg = tomllib.load(f)
    #     return cfg

    def __create_menubar(self) -> None:
        """创建菜单栏，菜单栏有四个内容分别为 `参数设置`、`股票价格预测`、`聊天求助`、`关于`

        Returns:
            None
        """
        memu_bar = tk.Menu(self.__root)
        memu_bar.add_command(label="股票信息搜索", command=self.__show_search)
        memu_bar.add_command(label="股票价格预测", command=self.__show_pred)
        memu_bar.add_command(label="聊天求助", command=self.__show_chat)
        memu_bar.add_command(label="关于", command=self.__show_about)

        self.__root.config(menu=memu_bar)

    def __show_search(self) -> None:
        """展示设置界面

        Returns:

        """
        # 先把其他页面消除，再显示本页内容
        self.__about_frame.pack_forget()
        self.__pred_frame.pack_forget()
        self.__chat_frame.pack_forget()
        self.__search_frame.pack()

    def __show_about(self) -> None:
        """展示关于界面

        Returns:
            None
        """
        # 先把其他页面消除，再显示本页内容
        self.__pred_frame.pack_forget()
        self.__chat_frame.pack_forget()
        self.__search_frame.pack_forget()
        self.__about_frame.pack()

    def __show_pred(self) -> None:
        """预测界面展示

        Returns:

        """
        # 先把其他页面消除，再显示本页内容
        self.__about_frame.pack_forget()
        self.__chat_frame.pack_forget()
        self.__search_frame.pack_forget()
        self.__pred_frame.pack()

    def __show_chat(self) -> None:
        """聊天界面展示

        Returns:

        """
        # 先把其他页面消除，再显示本页内容
        self.__pred_frame.pack_forget()
        self.__about_frame.pack_forget()
        self.__search_frame.pack_forget()
        self.__chat_frame.pack()
        # 请求用户名并聚焦至聊天界面
        self.__chat_frame.request_username()


if __name__ == '__main__':
    # 创建 `tkinter` 中的 `TK` 类，以实现窗口
    root = tk.Tk()
    MainPage(_root=root)
    root.mainloop()
