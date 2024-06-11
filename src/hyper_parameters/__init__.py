#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 3:13
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/hyper_parameters/__init__.py.py
# @Software: VS Code

__all__ = ["cfg"]

__description__ = ("一些超参数，因为打包成 `exe` 可执行文件不方便读取例如 `json` 或 `toml` 的一些外在超参数配置，所以这里直接将配置"
                   "写为字典存储在此处")

__author__ = "Zengxian Wang <zengxian822.wang@gmail.com>"
__version__ = "0.1"
__date__ = "10th June 2024"

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
        'track': False, 'wandb_project_name': 'Myriad'
    },
    'setting': {
        'multi_process': True, 'num_process': 8
    }
}
