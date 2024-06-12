#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 4:03
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/pred_frame.py
# @Software: VS Code

import toml
import torch
import time
import random

import baostock as bs
import tkinter as tk
import pandas as pd
import numpy as np
import lightning as pl

from tkinter import ttk
from tkinter import messagebox
from lightning.pytorch.loggers import TensorBoardLogger

from lstm_model import LstmModule, LitAutoLstm


class PredFrame(tk.Frame):
    """预测设置界面的包装实现

    """

    def __init__(self, _root: tk.Tk, _toml_path: str = "./config.toml"):
        # 执行父类的构造函数，使得我们能够调用父类的属性。
        super().__init__(_root)
        self.__toml_path = _toml_path
        # 设置存储股票代码的变量
        self.__index_code_holder = tk.StringVar()
        self.__status = tk.StringVar()
        self.__low_price = tk.StringVar()
        self.__high_price = tk.StringVar()
        self.__create_page()

    def __create_page(self):
        # 设置框
        tk.Label(self, text="所预测股票的代码（例如 000001 ）").grid(row=0, column=0)
        tk.Entry(self, textvariable=self.__index_code_holder).grid(row=0, column=1)

        # 设置下拉框 ddb 是 下拉框（drop down box） 的意思
        self.__stock_type_ddl = ttk.Combobox(self)
        self.__stock_type_ddl.grid(row=1, column=1)
        self.__stock_type_ddl['value'] = ['沪A', '深A']

        # 创造预测按钮
        self.__button = tk.Button(self, text='预测', command=self.__do_pred)
        self.__button.grid(row=3, column=1)
        tk.Label(self, textvariable=self.__status).grid(row=5, column=1)
        tk.Label(self, textvariable=self.__low_price).grid(row=7, column=1)
        tk.Label(self, textvariable=self.__high_price).grid(row=8, column=1)

    def __get_data(self):
        """通过 baostock 预测股票明天的最高价和最低价

        Args:
            self:
            index_code: 股票代码

        Returns:

        """
        # 这样才可以获得值
        self.__ddl_get = self.__stock_type_ddl.get()

        if self.__ddl_get == '深A':
            self.__index_code = 'sz.' + self.__index_code_holder.get()
        elif self.__ddl_get == '沪A':
            self.__index_code = 'sh.' + self.__index_code_holder.get()
        print(self.__index_code)

        # 训练的股票代码
        # 登录 baostock 系统
        self.__lg = bs.login()
        rs = bs.query_history_k_data_plus(self.__index_code,
                                          "code,date,open,close,low,high,volume,amount,pctChg",
                                          frequency="d", adjustflag="3")
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
        """预测触发的函数，具体就是讲 `src/lstm_model/pl_lstm.py` 中的测试实现内容搬过来

        Returns:

        """
        # 要作为特征（feature）的列，我们这里使用 `open`, `close`, `low`, `high`, `volume`, `money`, `change` 这些数据
        # 来进行预测分析，分别为 `开盘价`，`闭盘价`，`当日最低`，`当日最高`，`成交量`，`成交金额`，`变化`。也可以用 list 如
        # [2,4,6,8] 来进行设置，只使用其中的一部分特征值
        self.__feature_columns = list(range(2, 9))
        # 要预测的列序号，要预测的数据所在的列，按原数据的第一列为 0 开始计算,  我们在此想做的是同时预测数据的第 4, 5 列，
        # 即 `最低价` 和 `最高价`（`low`, `high`）
        self.__label_columns = [4, 5]
        # 获得超参数数据
        self.__args = toml.load(self.__toml_path)

        # 获得数据，其中数据可以通过 `train_data.values` 得到，特征的名称即列名可以通过 `train_data.columns.tolist()` 进行获取
        # 这里注意现在所有的项目都是 `object` 类型，我们必须要先将其转化为 `float` 类型才可以进行后续处理
        self.__train_data = self.__get_data().iloc[:, self.__feature_columns]
        self.__train_data.dropna(axis=0, how='any')
        # 使用 `pandas.to_numeric` 将所有数据转化为 `float` 形式
        for _cn in self.__train_data.columns:
            self.__train_data[_cn] = pd.to_numeric(self.__train_data[_cn], downcast='float')
            print(self.__train_data[_cn].dtypes)
        print(self.__train_data)

        # 设置保存位置的名称
        run_name = f"{self.__args["parameters"]["seed"]}__{int(time.time())}"

        # 设置随机数种子，保证每次都一样
        random.seed(self.__args["parameters"]["seed"])
        np.random.seed(self.__args["parameters"]["seed"])
        torch.manual_seed(self.__args["parameters"]["seed"])
        torch.cuda.manual_seed_all(self.__args["parameters"]["seed"])
        torch.backends.cudnn.deterministic = True

        # 设置记录 日志（log）用的 TensorBoard 组件
        # TensorBoard 是用来可视化模型信息的
        self.__tb_logger = TensorBoardLogger(
            save_dir=f"{self.__args["save_model"]["save_prefix"]}/res/stock-lstm/lightning_logs/{self.__index_code}",
            name="stock-lstm", default_hp_metric=False
        )

        # 初始化 `Trainer`，使用 Weights & Biases 去打印日志和可视化训练过程
        if self.__args["wandb"]["track"]:
            from lightning.pytorch.loggers import WandbLogger

            self.__wandb_logger = WandbLogger(
                project=self.__args["wandb"]["wandb_project_name"],
                entity=self.__args["wandb"]["wandb_entity"],
                save_dir=f"{self.__args["save_model"]["save_prefix"]}/res/stock-lstm/{self.__index_code}",
                sync_tensorboard=True,
                config=vars(self.__args),
                name=run_name
            )

            self.__trainer = pl.Trainer(
                max_epochs=self.__args["parameters"]["max_epochs"],
                default_root_dir=f"{self.__args["save_model"]["save_prefix"]}/res/stock-lstm/{self.__index_code}",
                logger=[self.__tb_logger, self.__wandb_logger]
            )
        else:
            self.__trainer = pl.Trainer(
                max_epochs=self.__args["parameters"]["max_epochs"],
                default_root_dir=f"{self.__args["save_model"]["save_prefix"]}/res/stock-lstm/{self.__index_code}",
                logger=[self.__tb_logger]
            )

        self.__auto_lstm = LitAutoLstm(
            LstmModule(len(self.__feature_columns), self.__args["parameters"]["hidden_size"], len(self.__label_columns),
                       self.__args["parameters"]["num_layers"], self.__args["parameters"]["dropout"]),
            self.__args, self.__train_data, self.__feature_columns, self.__label_columns
        )

        self.__trainer.fit(self.__auto_lstm)

        pred_result = self.__trainer.predict(self.__auto_lstm, ckpt_path="best")
        pred_result = np.squeeze(pred_result, axis=0)
        pred_result = np.concatenate(pred_result)
        print(pred_result.shape)

        # 进行尝试，如果预测失败则跳出信息
        try:
            self.__predict_data = pred_result * self.__auto_lstm.std[self.__auto_lstm.label_in_feature_index] + \
                                  self.__auto_lstm.mean[self.__auto_lstm.label_in_feature_index]

            if self.__predict_data[-1][0] == np.nan:
                # print("预测失败，请更换其他股票预测")
                # self.__status.set("预测失败，请更换其他股票预测")
                messagebox.showerror(title="Myriad v0.1.0", message="预测失败，请更换其他股票预测")
            else:
                # 预测成功就打印价格，保留 3 位小数
                self.__status.set("预测成功")
                self.__low_price.set(f"预测明日最低价格：{self.__predict_data[-1][0]:.3f}")
                self.__high_price.set(f"预测明日最高价格：{self.__predict_data[-1][1]:.3f}")
        except Exception:
            messagebox.showerror(title="Myriad v0.1.0", message="预测失败，请更换其他股票预测")


if __name__ == '__main__':
    pass
