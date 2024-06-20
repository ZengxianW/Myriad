#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/9 15:51
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/lstm_model/pl_lstm.py
# @Software: VS Code


# 该文件定义了一个基于 PyTorch Lightning 的 LSTM 模型，并实现了训练、验证、测试和预测的流程。
# 提供了数据预处理、模型初始化、优化器配置和日志记录功能。
# 包含主程序，用于加载数据、设置参数、训练模型和进行预测。

import torch

import pandas as pd
import numpy as np

import lightning.pytorch as pl

# 用于数据加载
from torch.utils.data import DataLoader, TensorDataset
# 用于划分训练和验证数据集
from sklearn.model_selection import train_test_split
from typing import Any

from lstm_model import LstmModule


class LitAutoLstm(pl.LightningModule):
    """使用 Pytorch Lightning 帮助简化训练过程，定义 nn.Modules 如何交互。

    """

    def __init__(self, _lstm: LstmModule, _args: dict, _df_data: pd.DataFrame,
                 _feature_columns: list, _label_columns: list):
        """初始化 LSTM 的 Pytorch Lightning 模型网络

        Args:
            _lstm: 传入的 LSTM 模型网络类
            _args: 传入的 配置定义，是  `dict` 类型
            _df_data: 传入的数据，包括训练集、验证集和测试集，默认为 `pd.DataFrame` 类型
            _feature_columns: 数据的特征列，`list` 类型，类似 [1, 2, 3] 的形式
            _label_columns: 要预测的数据所在的列，按原数据的第一列为 0 开始计算
        """
        super().__init__()
        self.__lstm = _lstm
        self.__args = _args
        self.__data = _df_data
        # 要作为特征（feature）的列，我们这里使用 `open`, `close`, `low`, `high`, `volume`, `money`, `change` 这些数据
        # 来进行预测分析，分别为 `开盘价`，`闭盘价`，`当日最低`，`当日最高`，`成交量`，`成交金额`，`变化`。也可以用 list 如
        # [2,4,6,8] 来进行设置，只使用其中的一部分特征值
        self.__feature_columns = _feature_columns
        # 要预测的数据所在的列，按原数据的第一列为 0 开始计算,  我们在此想做的是同时预测数据的第 4, 5 列，
        # 即 `最低价` 和 `最高价`（`low`, `high`）
        self.__label_columns = _label_columns

        # 我们需要进行归一化操作去量纲，即这些数据的大小不在同一个数值区间内，我们需要将他们这些特征集中到同一个数值区间内
        # 这里假设原始数据分布接近正态分布，所以使用零均值标准化（Z-score Normalization）将原始数据标准化为均值为 0 ，方差为 1 的分布。
        # 计算数据的均值
        self.mean = np.mean(self.__data, axis=0).to_numpy()
        # 计算数据的标准差
        self.std = np.std(self.__data, axis=0).to_numpy()
        # 得到的标准化之后的数据，公式 `norm = (x - \mu) / \sigma`，\mu 是平均值，\sigma 是标准差
        self.__norm_data = (self.__data - self.mean) / self.std
        # 数据的总量
        self.__data_num = self.__data.shape[0]
        # 作为训练数据集的数据的数量
        self.__train_num = int(
            self.__data_num * self.__args["parameters"]["train_data_rate"])

        # 我们需要找到我们所预测的这两个数据列，在总共的数据列中的序号，例如 [`open`, `close`, `low`, `high`, `volume`, `money`,
        # `change`] 这个列表，我们预测的数据 [`low`, `high`] 的序号为 [2, 3]，而他俩在原本的原始数据中的序号为 [4, 5]
        self.label_in_feature_index = (
            (lambda x, y: [x.index(i) for i in y])
            (self.__feature_columns, self.__label_columns)
        )

        # 选择日期为前 `train_data_rate` 的数据作为训练数据
        self.__feature_data = self.__norm_data[:self.__train_num]
        # 将后续几天的数据作为我们需要预测的内容，也就是 `label`，这里就是将之前的 `norm_data` 的数据往后移一天
        self.__label_data = self.__norm_data.iloc[
            self.__args["parameters"]["predict_day"]:
            self.__args["parameters"]["predict_day"] + self.__train_num,
            self.label_in_feature_index
        ]

        # 设置连续训练，在该模式下，每 `time_step` 行数据会作为一个样本，两个样本之间错开 `time_step` 行，
        # 比如：以 `time_step=20` 为例，最终实现结果为： 1-20 行，21-40行 ······ 一直到数据末尾，然后又是
        # 2-21 行，22-41 行 ······ 一直到数据的末尾最终是 19-59 行，22-41 行 ······ ，一直到数据末尾。这
        # 样才可以把上一个样本的最终状态 `final_state` 作为下一个样本的初始状态 `init_state`。由于股票预测
        # 的时序特征非常重要，训练数据必须按照时间排序，所以不能对数据集进行打乱操作（shuffle）。

        # `_train_x` 是特征数据，形状 (`data_num`, `time_step`, 7)，7 是一共有 7 个需要考虑的变量
        _train_x = [
            self.__feature_data[
                start_index + i * self.__args["parameters"]["time_step"]: start_index + (i + 1) *
                self.__args["parameters"]["time_step"]]
            for start_index in range(self.__args["parameters"]["time_step"])
            for i in range((self.__train_num - start_index) // self.__args["parameters"]["time_step"])
        ]
        # `_train_y` 是特征数据所对应预测的 `label` 值，例如 `train_x` 是第 1-20 天数据，`train_y` 是第 21
        # 天数据
        _train_y = [
            self.__label_data[
                start_index + i * self.__args["parameters"]["time_step"]: start_index + (i + 1) *
                self.__args["parameters"]["time_step"]]
            for start_index in range(self.__args["parameters"]["time_step"])
            for i in range((self.__train_num - start_index) // self.__args["parameters"]["time_step"])
        ]
        _train_x, _train_y = np.array(_train_x), np.array(_train_y)
        # 划分训练和验证集，并打乱；将训练数据中的 `valid_data_rate` 部分划归为验证集，剩下的才是真的训练集。
        # 借助 `sklearn` 中的 `train_test_split` 方法进行打乱
        _train_x, _valid_x, _train_y, _valid_y = train_test_split(
            _train_x, _train_y, test_size=self.__args["parameters"]["valid_data_rate"], shuffle=True
        )

        # 使用 `torch` 的 `DataLoader()` 方法，把数据装载，但首先需要转换成 `np.ndarray` 格式，之后在在类内函数
        # `train_dataloader()`，`val_dataloader()` 中使用 `DataLoader` 方法加载。
        self.__train_x, self.__train_y = torch.from_numpy(
            _train_x).float(), torch.from_numpy(_train_y).float()
        self.__valid_x, self.__valid_y = torch.from_numpy(
            _valid_x).float(), torch.from_numpy(_valid_y).float()

        self.__start_num_in_test = None

# 定义训练步骤，计算训练集的损失值。
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """ `training_step` 定义整个训练过程中每一次循环所需要做的事情，并且返回每一步的 loss 值

        Args:
            batch:
            batch_idx:

        Returns:
            loss: 训练集的每一步的 loss 值
        """
        x, y = batch
        pred_y, hidden_train = self.__lstm(x)
        # 设定评判标准（criterion）为均方误差（mean squared error, MSE）
        criterion = torch.nn.MSELoss()
        # 将预测的结果与实际结果进行对比，计算他们两个之间的 MSE 作为深度模型的 loss 值
        loss = criterion(pred_y, y)
        # 使用 wandb 或者 tensorboard
        self.log_dict({"train_loss": loss.item()})
        return loss

# 定义验证步骤，计算验证集的损失值。
    def validation_step(self, batch, batch_idx: int) -> None:
        """ `validation_step` 定义整个验证过程中每一次循环所需要做的事情，并且返回每一步的 loss 值

        Args:
            batch:
            batch_idx:

        Returns:

        """
        x, y = batch
        pred_y, hidden_train = self.__lstm(x)
        # 设定评判标准（criterion）为均方误差（mean squared error, MSE）
        criterion = torch.nn.MSELoss()
        # 将预测的结果与实际结果进行对比，计算他们两个之间的 MSE 作为深度模型的 loss 值
        loss = criterion(pred_y, y)
        # 使用 wandb 或者 tensorboard
        self.log_dict({"val_loss": loss.item()})

# 定义测试步骤，计算测试集的损失值
    def test_step(self, batch, batch_idx: int) -> None:
        """ `test_step` 定义整个测试过程中每一次循环所需要做的事情，并且返回每一步的 loss 值

        Args:
            batch:
            batch_idx:

        Returns:

        """
        x, y = batch
        pred_y, hidden_train = self.__lstm(x)
        # 设定评判标准（criterion）为均方误差（mean squared error, MSE）
        criterion = torch.nn.MSELoss()
        # 将预测的结果与实际结果进行对比，计算他们两个之间的 MSE 作为深度模型的 loss 值
        loss = criterion(pred_y, y)
        # 使用 wandb 或者 tensorboard
        self.log_dict({"test_loss": loss.item()})

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """ 在这一步定义模型的优化器

        Returns:
            optimizer: 优化器
        """
        # 设定优化器为 `Adam`，其中 `lr` 代表学习率，`eps` 代表 `Adam` 中提高数值稳定性的一个参数
        optimizer = torch.optim.Adam(self.parameters(), lr=self.__args["parameters"]["learning_rate"],
                                     eps=0.01 / self.__args["parameters"]["batch_size"])
        return optimizer


    def train_dataloader(self) -> DataLoader:
        """使用 `train_dataloader()` 方法生成训练数据加载器

        Returns:
            _train_loader: 以 `torch.utils.data.DataLoader` 类型表示的训练集
        """
        _train_loader = DataLoader(
            TensorDataset(self.__train_x, self.__train_y),
            batch_size=self.__args["parameters"]["batch_size"]
        )
        return _train_loader

    def val_dataloader(self) -> DataLoader:
        """使用 `train_dataloader()` 方法生成验证集数据加载器

        Returns:
            _valid_loader: 以 `torch.utils.data.DataLoader` 类型表示的验证集
        """
        _valid_loader = DataLoader(
            TensorDataset(self.__valid_x, self.__valid_y),
            batch_size=self.__args["parameters"]["batch_size"]
        )
        return _valid_loader

    def test_dataloader(self) -> DataLoader:
        """使用 `test_dataloader()` 方法生成验证集数据加载器

        Returns:
            _test_loader: 以 `torch.utils.data.DataLoader` 类型表示的预测数据集
        """
        _feature_data = self.__norm_data[self.__train_num - 1:-1]
        # 防止 `time_step` 大于测试集数量
        _sample_interval = min(
            _feature_data.shape[0], self.__args["parameters"]["time_step"])
        # 这些天的数据不够一个 `sample_interval`
        self.__start_num_in_test = _feature_data.shape[0] % _sample_interval
        _time_step_size = _feature_data.shape[0] // _sample_interval

        # 在测试数据中，每 `time_step` 行数据会作为一个样本，两个样本错开 `time_step` 行
        # 比如：1-20 行，21-40 行 ······ 到数据末尾。
        _test_x = torch.from_numpy(np.array([
            _feature_data[
                self.__start_num_in_test + i * _sample_interval - 1:
                self.__start_num_in_test + (i + 1) * _sample_interval - 1
            ] for i in range(_time_step_size)
        ])).float()
        _test_y = torch.from_numpy(np.array([
            _feature_data[
                self.__start_num_in_test + i * _sample_interval:
                self.__start_num_in_test + (i + 1) * _sample_interval
            ] for i in range(_time_step_size)
        ])).float()
        _test_loader = DataLoader(
            TensorDataset(_test_x, _test_y),
            batch_size=self.__args["parameters"]["batch_size"]
        )
        return _test_loader

    def predict_dataloader(self) -> DataLoader:
        """使用 `tpredict_dataloader()` 方法生成验证集数据加载器

        Returns:
            _pred_loader: 以 `torch.utils.data.DataLoader` 类型表示的预测数据集
        """
        _feature_data = self.__norm_data[self.__train_num - 1:-1]
        # 防止 `time_step` 大于测试集数量
        _sample_interval = min(
            _feature_data.shape[0], self.__args["parameters"]["time_step"])
        # 这些天的数据不够一个 `sample_interval`
        self.__start_num_in_test = _feature_data.shape[0] % _sample_interval
        _time_step_size = _feature_data.shape[0] // _sample_interval

        # 在测试数据中，每 `time_step` 行数据会作为一个样本，两个样本错开 `time_step` 行
        # 比如：1-20 行，21-40 行 ······ 到数据末尾。
        _pred_x = torch.from_numpy(np.array([
            _feature_data[
                self.__start_num_in_test + i * _sample_interval: self.__start_num_in_test + (i + 1) * _sample_interval
            ] for i in range(_time_step_size)
        ])).float()

        # 设置预测用的数据集 `DataLoader()`，其中 `num_workers` 参数表示同时进行的数量。
        _pred_loader = DataLoader(
            TensorDataset(_pred_x),
            # batch_size=1
            batch_size=self.__args["parameters"]["batch_size"],
            # num_workers=_time_step_size
        )
        return _pred_loader

# 定义预测步骤，返回预测结果
    def predict_step(self, batch, batch_idx: int) -> Any:
        """模型进行预测的时候使用的步骤

        Args:
            batch:
            batch_idx:

        Returns:
            result: 预测结果
        """
        x = batch[0]
        pred_y, _ = self.__lstm(x)
        cur_pred = torch.squeeze(pred_y, dim=0)
        # 先去梯度信息，如果在 gpu 要转到 cpu，最后要返回 `numpy` 数据
        _pred_result = cur_pred.detach().cpu().numpy()
        return _pred_result


if __name__ == '__main__':

    import time
    import random

    from lightning.pytorch.loggers import TensorBoardLogger

    cfg = {
        'parameters': {
            'seed': 1, 'train_data_rate': 0.95, 'valid_data_rate': 0.15, 'predict_day': 1, 'time_step': 50,
            'debug_num': 500, 'learning_rate': 0.0001, 'batch_size': 128, 'hidden_size': 128, 'num_layers': 2,
            'dropout': 0.2, 'max_epochs': 128
        },
        'save_model': {
            'save_model': True, 'save_prefix': './'
        },
        'wandb': {
            'track': True, 'wandb_project_name': 'Myriad', "wandb_entity": None
        }
    }

    # def parse_args():
    #     """使用 `argparse` 库设置需要用到的超参数。
    #
    #     Returns:
    #         args: 超参数，类型为 `argparse.ArgumentParser()`
    #     """
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument("--seed", type=int, default=1,
    #                         help="seed of the experiment.")
    #
    #     # 与深度学习模型有关的参数
    #     parser.add_argument("--train_data_rate", type=float, default=0.95,
    #                         help="训练数据占总体数据比例，测试数据就是 `1 - train_data_rate`。")
    #     parser.add_argument("--valid_data_rate", type=float, default=0.15,
    #                         help="验证数据占训练数据比例，验证集在训练过程使用，用于验证该机器学习算法稳定性和可靠性。")
    #     parser.add_argument("--predict_day", type=int, default=1,
    #                         help="描述这个模型想要预测未来 `predict_day` 天的数据。")
    #     parser.add_argument("--time_step", type=int, default=20,
    #                         help="设置用前 `time_step` 天的数据来预测，也是 LSTM 使用的 time step 数，一定要保证训练数据量大于它。")
    #     parser.add_argument("--debug_num", type=int, default=500,
    #                         help="测试情况，在测试数据中，使用 `debug_num` 条数据进行检验测试。")
    #     parser.add_argument("--learning_rate", type=float, default=1e-4,
    #                         help="优化器（optimizer）的学习率。")
    #     parser.add_argument("--batch_size", type=int, default=128,
    #                         help="深度学习模型训练的 batch size。")
    #     parser.add_argument("--hidden_size", type=int, default=128,
    #                         help="LSTM 的隐藏层大小，也是其输出大小。")
    #     parser.add_argument("--num_layers", type=int, default=2,
    #                         help="LSTM 的堆叠层数。")
    #     parser.add_argument("--dropout", type=float, default=0.2,
    #                         help="dropout 的大小。")
    #     parser.add_argument("--max_epochs", type=int, default=16,
    #                         help="在没有提前停止的情况下，整个训练最多被悬链的最大次数")
    #
    #     # 与保存模型，可视化有关的参数
    #     parser.add_argument("--save_model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    #                         help="是否将训练过的模型参数进行保存。")
    #     parser.add_argument("--save_prefix", type=str, default="../..",
    #                         help="保存模型参数的位置前缀。")
    #     parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=False,
    #                         help="如果触发，这个训练将会被 Weights and Biases 所追踪。")
    #     parser.add_argument("--wandb_project_name", type=str, default="Myriad",
    #                         help="wandb 的项目名。")
    #     parser.add_argument("--wandb_entity", type=str, default=None,
    #                         help="wandb 项目的实体 entity (团队 team)。")
    #
    #     args = vars(parser.parse_args())
    #
    #     return args

    # `distutils.util.strtobool` 在 python3.12 中被迫废弃，这里直接借助它的代码，因为
    # `argparse` 库添加参数的类型不能直接设置为 `type=bool`，而是需要用 `type=lambda x: bool(strtobool(x))`

    def strtobool(val: str):
        """Convert a string representation of truth to true (1) or false (0).
        True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
        are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
        'val' is anything else.
        """
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return 1
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return 0
        else:
            raise ValueError("invalid truth value %r" % (val,))

    # 训练的股票代码
    index_code = "stock_data"
    # 训练数据存放的位置（训练数据使用 `csv` 文件格式进行存储的）
    train_data_path = f"./data/{index_code}.csv"
    # 测试情况，在测试数据中，使用 `debug_num` 条数据进行检验测试
    # debug_num = 500
    # 要作为特征（feature）的列，我们这里使用 `open`, `close`, `low`, `high`, `volume`, `money`, `change` 这些数据
    # 来进行预测分析，分别为 `开盘价`，`闭盘价`，`当日最低`，`当日最高`，`成交量`，`成交金额`，`变化`。也可以用 list 如
    # [2,4,6,8] 来进行设置，只使用其中的一部分特征值
    feature_columns = list(range(2, 9))
    # 要预测的列序号，要预测的数据所在的列，按原数据的第一列为 0 开始计算,  我们在此想做的是同时预测数据的第 4, 5 列，
    # 即 `最低价` 和 `最高价`（`low`, `high`）
    label_columns = [4, 5]

    # 获得数据，其中数据可以通过 `data.values` 得到，特征的名称即列名可以通过 `data.columns.tolist()` 进行获取
    # data = pd.read_csv(train_data_path, nrows=debug_num, usecols=feature_columns)
    data = pd.read_csv(train_data_path, usecols=feature_columns)
    # 打印数据出来看一下
    print(data.head())

    # 获得超参数数据
    # args = parse_args()
    args = cfg

    # 设置保存位置的名称
    run_name = f"{args["parameters"]["seed"]}__{int(time.time())}"

    # 设置随机数种子，保证每次都一样
    random.seed(args["parameters"]["seed"])
    np.random.seed(args["parameters"]["seed"])
    torch.manual_seed(args["parameters"]["seed"])
    torch.cuda.manual_seed_all(args["parameters"]["seed"])
    torch.backends.cudnn.deterministic = True

    # 设置使用 `TensorBoard` 的日志保存器
    tb_logger = TensorBoardLogger(
        save_dir=f"{args["save_model"]["save_prefix"]
                    }/res/stock-lstm/lightning_logs/{index_code}",
        name="stock-lstm", default_hp_metric=False
    )

    # 初始化 `Trainer`
    if args["wandb"]["track"]:
        from lightning.pytorch.loggers import WandbLogger

        # 设置使用 `wandb` 的日志保存器
        wandb_logger = WandbLogger(
            project=args["wandb"]["wandb_project_name"],
            entity=args["wandb"]["wandb_entity"],
            save_dir=f"{args["save_model"]["save_prefix"]
                        }/res/stock-lstm/{index_code}",
            sync_tensorboard=True,
            config=args,
            name=run_name
        )

        trainer = pl.Trainer(
            max_epochs=args["parameters"]["max_epochs"],
            default_root_dir=f"{
                args["save_model"]["save_prefix"]}/res/stock-lstm/{index_code}",
            logger=[tb_logger, wandb_logger]
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args["parameters"]["max_epochs"],
            default_root_dir=f"{
                args["save_model"]["save_prefix"]}/res/stock-lstm/{index_code}",
            logger=[tb_logger]
        )

    auto_lstm = LitAutoLstm(
        LstmModule(len(feature_columns), args["parameters"]["hidden_size"], len(label_columns),
                   args["parameters"]["num_layers"], args["parameters"]["dropout"]),
        args, data, feature_columns, label_columns
    )

    trainer.fit(auto_lstm)

    pred_result = trainer.predict(auto_lstm, ckpt_path="best")
    pred_result = np.squeeze(pred_result, axis=0)
    pred_result = np.concatenate(pred_result)
    # print(pred_result)

    print(auto_lstm.mean[auto_lstm.label_in_feature_index].shape,
          type(auto_lstm.mean[auto_lstm.label_in_feature_index]))
    print(auto_lstm.std[auto_lstm.label_in_feature_index].shape,
          type(auto_lstm.std[auto_lstm.label_in_feature_index]))
    print(pred_result.shape, type(pred_result))

    # 通过保存的均值和方差还原数据
    predict_data = pred_result * auto_lstm.std[auto_lstm.label_in_feature_index] + \
        auto_lstm.mean[auto_lstm.label_in_feature_index]

    print(predict_data)
