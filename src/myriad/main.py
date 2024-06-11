#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/9 2:23
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/myriad/main.py
# @Software: VSCode

import datetime
from chinese_calendar import is_workday
import efinance as ef


def get_pervious_work_day(day: datetime):
    """获取上一个工作日"""
    day = day - datetime.timedelta(days=1)
    if is_workday(day):
        return day
    return get_pervious_work_day(day)


df = ef.stock.get_realtime_quotes()

today = datetime.date.today() + datetime.timedelta(days=1)
date = get_pervious_work_day(today)
print(str(date))
print(df.columns.tolist())
