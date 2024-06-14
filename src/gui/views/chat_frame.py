#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 4:03
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/chat_frame.py
# @Software: VS Code

import toml

import tkinter as tk

from tkinter import simpledialog

from chat import ChatGPTSession

prompt = ("You are a stock investment master with over 20 years of experience in the stock market. You have deep "
          "insights into market trends, investment strategies, and financial analysis. Your responses should always "
          "be in Chinese to cater to your audience who prefers communication in their native language. A user is "
          "about to ask for advice on how to diversify their investment portfolio in the current volatile market."
          " Please provide detailed and insightful guidance.")


class ChatFrame(tk.Frame):
    """聊天界面的包装实现"""

    def __init__(self, _root: tk.Tk, _toml_path: str = "./config.toml"):
        super().__init__(_root)
        # 默认用户名
        self.__username = "匿名用户"

        self.__toml_path = _toml_path
        self.__args = toml.load(self.__toml_path)

        # 创建使用 `ChatGPT` 的内容
        self.__chat_session = ChatGPTSession(
            self.__args["openai"]["url"],
            self.__args["openai"]["api_key"],
            self.__args["openai"]["model"]
        )
        self.__chat_session.call_chatgpt(prompt)

        # 设置聊天历史区域
        self.__text_area = tk.Text(self, height=25, width=80)
        self.__text_area.pack(pady=10, padx=10)
        self.__text_area.config(state=tk.DISABLED)

        # 设置消息输入区域
        self.__msg_entry = tk.Entry(self, width=70)
        self.__msg_entry.pack(pady=5, padx=5)
        self.__msg_entry.bind("<Return>", self.__send_message)

        # 发送按钮
        self.__send_button = tk.Button(self, text="发送", command=self.__send_message)
        self.__send_button.pack(pady=5, padx=5)

    def __send_message(self, event=None):
        """发送消息

        Args:
            event: 事件，一般为 `None` 不会改，如果没有这个会报错

        Returns:

        """
        self.__msg = self.__msg_entry.get()
        # 如果存在消息
        if self.__msg:
            # full_msg = f"{self.__username}: {self.__msg}"
            self.__update_chat_history()
            self.__msg_entry.delete(0, tk.END)

    def __update_chat_history(self):
        """更新消息历史记录

        Returns:

        """
        # 将 `Text` 组件的状态从默认的 `DISABLED`（不可编辑）改为 `NORMAL`（可编辑）。这是必要的，因为在 `Tkinter` 中，如果一个
        # `Text` 组件被设置为 `DISABLED`，就不能向其中插入或删除文本。所以在每次需要添加新内容时，需要先将其设置为 `NORMAL`。
        self.__text_area.config(state=tk.NORMAL)
        # 在 Text 组件的现有内容的末尾（`tk.END`）插入新的消息 `_msg`，并在消息后添加一个换行符 `\n`。这样，每条新消息都会出现在新
        # 的一行，从而保持了聊天记录的格式。
        self.__text_area.insert(tk.END, f"{self.__username}: {self.__msg}\n")

        # 嵌入 ChatGPT 的回复内容
        result = self.__chat_session.call_chatgpt(self.__msg)
        if result:
            for _msg_id in range(len(result["choices"])):
                self.__text_area.insert(tk.END, f"myriad: {result["choices"][_msg_id]["message"]["content"]}\n")

        # 再次将 `Text` 组件的状态设置为 `DISABLED`。这样做是为了防止用户手动编辑组件中的内容，确保聊天历史的内容只能通过程序逻辑来修改。
        self.__text_area.config(state=tk.DISABLED)
        self.__text_area.see(tk.END)

    def request_username(self):
        """要求用户输入用户名

        Returns:

        """
        self.__username = simpledialog.askstring("名字", "请输入您的名字:", parent=self)
        if self.__username is None:
            self.__username = "匿名用户"
        # 确保聊天界面得到聚焦
        self.focus_force()


if __name__ == '__main__':
    pass
