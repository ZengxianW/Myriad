#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 6:33
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/pred/get_data.py
# @Software: VS Code

if __name__ == '__main__':
    import tomllib

    with open("../../config.toml", "rb") as f:
        cfg = tomllib.load(f)
    print(cfg["parameters"])


