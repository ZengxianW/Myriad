#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/11 4:03
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/gui/chat_frame.py
# @Software: VS Code


# 这个 chat_frame.py 文件实现了一个基于 tkinter 的图形用户界面，用于与 ChatGPT 进行聊天。
# 用户可以通过界面输入消息并获取 ChatGPT 的回复。
# 整个系统通过配置文件加载必要参数，使用 ChatGPTSession 进行会话，并显示在图形界面中。


# 导入 toml 模块，用于处理 TOML 配置文件
import toml

# 导入 tkinter 模块，用于创建图形用户界面
import tkinter as tk
# 导入 simpledialog 模块，用于简单的对话框交互。
from tkinter import simpledialog
# 导入自定义的 ChatGPTSession 类，用于与 ChatGPT 进行会话。
from chat import ChatGPTSession

# 定义一个 prompt，用于初始化 ChatGPT 会话。
prompt = ("You are a stock investment master with over 20 years of experience in the stock market. You have deep "
          "insights into market trends, investment strategies, and financial analysis. Your responses should always "
          "be in Chinese to cater to your audience who prefers communication in their native language. A user is "
          "about to ask for advice on how to diversify their investment portfolio in the current volatile market."
          " Please provide detailed and insightful guidance.")


# 定义了一个继承自 tk.Frame 的类 ChatFrame，表示一个聊天界面
class ChatFrame(tk.Frame):
    """聊天界面的包装实现"""

# 类的构造函数，接收一个根窗口 _root 作为参数
    def __init__(self, _root: tk.Tk, _toml_path: str = "./config.toml"):
        super().__init__(_root)
        # 默认用户名
        self.__username = "匿名用户"

        self.__toml_path = _toml_path
        self.__args = toml.load(self.__toml_path)

        # 创建使用 `ChatGPT` 的内容
        # 使用配置文件中的参数创建一个 ChatGPTSession 对象，并使用预设提示 prompt 初始化会话。
        self.__chat_session = ChatGPTSession(
            self.__args["openai"]["url"],
            self.__args["openai"]["api_key"],
            self.__args["openai"]["model"]
        )
        self.__chat_session.call_chatgpt(prompt)

        # 设置聊天历史区域
        # 创建一个 Text 组件，用于显示聊天历史记录，并设置其不可编辑状态。
        self.__text_area = tk.Text(self, height=25, width=80)
        self.__text_area.pack(pady=10, padx=10)
        self.__text_area.config(state=tk.DISABLED)

        # 设置消息输入区域
        # 创建一个 Entry 组件，用于输入消息，并绑定回车键事件触发 
        self.__msg_entry = tk.Entry(self, width=70)
        # 使用 pack() 方法将输入框、发送按钮、滚动条和输出框添加到界面中，并设置它们的位置和填充方式。
        self.__msg_entry.pack(pady=5, padx=5)
        self.__msg_entry.bind("<Return>", self.__send_message)

        # 发送按钮
        # 创建一个按钮，点击时调用 __send_message 方法发送消息
        self.__send_button = tk.Button(self, text="发送", command=self.__send_message)
        self.__send_button.pack(pady=5, padx=5)

# 定义 __send_message 方法，获取用户输入的消息，清空输入框，将消息添加到历史记录，并获取 ChatGPT 的响应，将响应添加到历史记录，并在输出框中显示。
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

# 方法用于更新聊天历史记录，将用户的消息和ChatGPT的回复显示在 Text 组件中
    def __update_chat_history(self):
        """更新消息历史记录

        Returns:

        """
        # 将 `Text` 组件的状态从默认的 `DISABLED`（不可编辑）改为 `NORMAL`（可编辑）。这是必要的，因为在 `Tkinter` 中，如果一个
        # `Text` 组件被设置为 `DISABLED`，就不能向其中插入或删除文本。所以在每次需要添加新内容时，需要先将其设置为 `NORMAL`。
        # 将 Text 组件的状态设置为可编辑
        self.__text_area.config(state=tk.NORMAL)
        # 在 Text 组件的现有内容的末尾（`tk.END`）插入新的消息 `_msg`，并在消息后添加一个换行符 `\n`。这样，每条新消息都会出现在新
        # 的一行，从而保持了聊天记录的格式。
        # 插入用户的消息
        self.__text_area.insert(tk.END, f"{self.__username}: {self.__msg}\n")

        # 嵌入 ChatGPT 的回复内容
        result = self.__chat_session.call_chatgpt(self.__msg)
        if result:
            for _msg_id in range(len(result["choices"])):
                self.__text_area.insert(tk.END, f"myriad: {result["choices"][_msg_id]["message"]["content"]}\n")

        # 再次将 `Text` 组件的状态设置为 `DISABLED`。这样做是为了防止用户手动编辑组件中的内容，确保聊天历史的内容只能通过程序逻辑来修改。
        # 将 Text 组件的状态设置为不可编辑
        self.__text_area.config(state=tk.DISABLED)
        # 滚动到最新消息
        self.__text_area.see(tk.END)

# request_username 方法弹出一个对话框，要求用户输入用户名。如果用户未输入，则默认使用 "匿名用户"
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
