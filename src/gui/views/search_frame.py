#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 4:12
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/views/search_frame.py
# @Software: VS Code

import datetime

import tkinter as tk

import efinance as ef

from tkinter import ttk
from tkinter import messagebox
from chinese_calendar import is_workday


def get_pervious_work_day(day: datetime):
    """获取上一个工作日

    Args:
        day: 需要判断的日期

    Returns:

    """

    day = day - datetime.timedelta(days=1)
    # 如果前一天是工作日，则直接返回
    if is_workday(day):
        return day
    # 否则再判断前一天的前一天
    return get_pervious_work_day(day)


def get_data():
    efdf = ef.stock.get_realtime_quotes()

    now = datetime.datetime.now().time()

    # 股市开盘时间 9:30，如果超过 9:30，则判断它前一天更新，否则判断今天更新
    # 这一步的目的是去掉已经退市的无用股票
    if datetime.time(9, 30) > now:
        today = datetime.date.today()
        workday = get_pervious_work_day(today)
        return efdf[efdf["最新交易日"] == str(workday)]
    else:
        today = datetime.date.today() + datetime.timedelta(days=1)
        workday = get_pervious_work_day(today)
        return efdf[efdf["最新交易日"] == str(workday)]


class SearchFrame(tk.Frame):
    """股票查找保存界面的包装实现

    """

    def __init__(self, _root):
        # 执行父类的构造函数，使得我们能够调用父类的属性。
        super().__init__(_root)
        # 创建一个滚动条控件，默认为垂直方向
        self.__vertical_sbar = tk.Scrollbar(self)
        # 将滚动条放置在右侧，并设置当窗口大小改变时滚动条会沿着垂直方向延展
        self.__vertical_sbar.pack(side=tk.RIGHT, fill=tk.Y)
        # 创建水平滚动条，默认为水平方向,当拖动窗口时会沿着X轴方向填充
        self.__horizontal_sbar = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.__horizontal_sbar.pack(side=tk.BOTTOM, fill=tk.X)

        # 通过 `efinance` 获取股票数据
        self.__efdf = get_data()
        # 获取列名
        self.__efdf_columns = self.__efdf.columns.tolist()

        # 创建展示表格组件
        self.__tree_view = ttk.Treeview(
            self,
            # height=15,  # 表格显示的行数
            columns=self.__efdf_columns,  # 显示的列
            show='headings',  # 隐藏首列
        )
        # 填充数据
        self.__create_treeview()
        # 放置窗口
        self.__tree_view.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # 使用 command 关联控件的 yview、xview方法
        self.__vertical_sbar.config(command=self.__tree_view.yview)
        self.__horizontal_sbar.config(command=self.__tree_view.xview)

        # 创造保存按钮
        self.__button = tk.Button(self, text='保存数据', command=self.__to_csv)
        self.__button.pack(side=tk.BOTTOM)

    def __create_treeview(self):
        """创建展示窗口组件

        Returns:

        """
        # print(self.__efdf)
        # 对展示表格组件填充数据
        for _x in self.__efdf:
            self.__tree_view.heading(_x, text=_x)
            self.__tree_view.column(_x, width=100)
        for _i in range(len(self.__efdf)):
            self.__tree_view.insert(
                '', _i, values=self.__efdf.iloc[_i, :].tolist())

    def __create_button(self):
        """创造按钮

        Returns:

        """
        button = tk.Button(self, text='保存数据')
        return button

    def __to_csv(self):
        """将数据保存到 `csv` 文件中

        Returns:

        """
        if not os.path.exists("./res"):
            os.makedirs("./res")
        self.__efdf.to_csv("./res/ef_data.csv", index=False)
        messagebox.showinfo(title="Myriad v0.1.0", message="保存成功")


if __name__ == '__main__':
    pass
