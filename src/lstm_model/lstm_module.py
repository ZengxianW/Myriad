#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/9 2:23
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/lstm_model/lstm_module.py
# @Software: VSCode


import torch.nn as nn

from torch import Tensor
from typing import Tuple, Optional


class LstmModule(nn.Module):
    r"""使用人工智能库 PyTorch 构造的长短时记忆（LSTM）网络模型

    """

    def __init__(self, _input_size: int, _hidden_size: int, _output_size: int,
                 _num_layers: int = 1, _dropout: float = 0., _batch_first: bool = True):
        r"""`__init__.py` 是类（class）的构造方法，在使用类创建对象之后被执行，
        用于给新创建的对象初始化属性。所有参数可以参考 Pytorch 的 LSTM 类实现，
        ref: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        Args:
            _input_size: 输入 `x` 中预期特征的数量
            _hidden_size: 隐藏状态 `h` 的特征数量
            _output_size: 最终输出的数据数量
            _num_layers: 循环的层数，E.g. 设置 ``num_layers=2`` 将会有两个 LSTM 堆叠在一起，
                形成一个 `堆叠 LSTM`（`stacked LSTM`）。第二个 LSTM 接收第一个 LSTM 的输出
                和计算的结果。Default: 1
            _dropout: 如果非零，则在每个输出上引入一个 `Dropout` 层
                LSTM 层除最后一层外，其 dropout 的概率为
                :attr:`dropout`. Default: 0
            _batch_first: 如果是 `True`，则输入和输出的 Tensor 将会以 `(batch, seq, feature)`
                的形式提供，而非 `(seq, batch, feature)` 形式
        """
        super(LstmModule, self).__init__()
        self.__lstm = nn.LSTM(input_size=_input_size, hidden_size=_hidden_size,
                              num_layers=_num_layers, batch_first=_batch_first, dropout=_dropout)
        self.__linear = nn.Linear(in_features=_hidden_size, out_features=_output_size)

    def forward(self, x: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) \
            -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Pytorch 框架构造函数中必须实现的前向传播放方式的实现函数

        Args:
            x: 需要输入 LSTM 的数据
            hx: 隐藏状态向量，但此处我们不需要，但是实现的时候必须要实现 `LSTM` 对应的 `forward` 函数。
                故此处必须有一个对应的参数。默认为两个参数组合成的元组，形式类似 (`h0`, `c0`)。其中 `h0`
                是初始隐藏状态（initial hidden state）。对于每个时间步的输入，LSTM 都会维护一个隐藏状
                态，`h0` 是在第一个时间步之前传递给 LSTM 的隐藏状态。而 `c0` 是初始细胞状态
                （initial cell state）。LSTM 的细胞状态是用来存储长期记忆的，它在每个时间步也会被更新，
                `c0` 是在第一个时间步之前传递给 LSTM 的细胞状态。 Default: None

        Returns:
            linear_out: 经过神经网络 `LSTM` 之后再经过一个线性变换 `Linear` 的输出。
            hx: 隐藏层数据，以 (`hn`, `cn`) 的状态输出，`hn` 是最后一个时间步的隐藏状态，形状与 `h0`
            相同。 `cn` 是最后一个时间步的细胞状态，形状与 `c0` 相同。
        """
        lstm_out, hx = self.__lstm(x, hx)
        linear_out = self.__linear(lstm_out)
        return linear_out, hx


# 测试这个文件中的模块是否能够运行
if __name__ == '__main__':
    # 测试 LstmNet 类
    import torch

    rnn = LstmModule(10, 20, 2, 2)
    inp = torch.randn(5, 3, 10)
    h0 = torch.randn(2, 5, 20)
    c0 = torch.randn(2, 5, 20)
    output, (hn, cn) = rnn(inp, (h0, c0))
    print(output, (hn, cn))
