#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 4:03
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/pred_frame.py
# @Software: VS Code

import baostock as bs
import tkinter as tk
import pandas as pd

from tkinter import ttk
from tkinter import messagebox


class PredFrame(tk.Frame):
    """预测设置界面的包装实现

    """

    def __init__(self, _root):
        # 执行父类的构造函数，使得我们能够调用父类的属性。
        super().__init__(_root)
        # 设置存储股票代码的变量
        self.index_code = tk.StringVar()
        self.__create_page()

    def __create_page(self):
        # 设置框
        tk.Label(self, text="所预测股票的代码（例如 300338 ）").grid(row=0, column=0)
        tk.Entry(self, textvariable=self.index_code).grid(row=0, column=1)
        # tk.Button(self,text="预测",command=self.__do_pred).grid(row=1, column=1)


        # 设置下拉框 ddb 是 下拉框（drop down box） 的意思
        self.stock_type_ddl = ttk.Combobox(self)
        self.stock_type_ddl.grid(row=1, column=1)
        self.stock_type_ddl['value'] = ['沪A', '深A']
        self.ddl_get = self.stock_type_ddl.get()

        # print(self.index_code.get())
        # print(self.ddl_get)
        # self.index_code = self.index_code.get()
        #
        # 创造预测按钮
        self.button = tk.Button(self, text='预测', command=self.__do_pred)
        self.button.grid(row=3, column=1)

        # self.__train_data = self.__get_data(index_code)


    def __get_data(self):
        """通过 baostock 预测股票明天的最高价和最低价

        Args:
            self:
            index_code: 股票代码

        Returns:

        """
        if self.ddl_get == '深A':
            self.index_code = 'sz.' + self.index_code
        elif self.ddl_get == '沪A':
            self.index_code = 'sh.' + self.index_code
        print(self.index_code)
        # 训练的股票代码
        # 登录 baostock 系统
        self.__lg = bs.login()
        rs = bs.query_history_k_data_plus(self.index_code,
                                          "code,date,open,close,low,high,volume,amount,pctChg",
                                          start_date='2021-01-01', frequency="d", adjustflag="3")
        # 登出 baostock 系统
        bs.logout()

        data_list = []
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            r = rs.get_row_data()
            data_list.append(r)
        train_data = pd.DataFrame(data_list, columns=rs.fields)
        train_data = train_data.rename(
            columns={'code': 'index_code', 'pctChg': 'change', 'amount': 'money'})
        if train_data.shape[0] <= 100:
            messagebox.showerror(title="Myriad v0.1.0", message="数据不足，所预测的股票需要至少上市 100 天以上。")

        return train_data

    def __do_pred(self):
        # self.__train_data = self.__get_data()
        print(self.ddl_get)
        # print(type(self.stock_type_ddl.get()))
        # 这样才可以获得值
        a=self.stock_type_ddl.get()
        print(a)
        print(type(self.index_code.get()))
