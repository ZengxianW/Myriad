#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2024/6/14 10:44
# @Author  : Zengxian Wang
# @Email   : zengxian822.wang@gmail.com
# @File    : src/chat/chatgpt_session.py
# @Software: VS Code
import requests


class ChatGPTSession:
    def __init__(self, _url: str, _api_key: str, _model: str):
        # API 端点地址
        self.__url = f"{_url}/v1/chat/completions"
        self.__api_key = _api_key
        self.__model = _model
        # 初始化一个空列表来存储历史记录
        self.__history = []

    def call_chatgpt(self, _prompt):
        """调用 chatgpt 的函数

        Args:
            _prompt: `prompt` 就是你对 chatgpt 说的话

        Returns:

        """
        headers = {
            "Authorization": f"Bearer {self.__api_key}",
            "Content-Type": "application/json"
        }

        # 构建包含所有历史消息的请求体
        messages = [{"role": "user", "content": message} for message in self.__history]
        # 添加当前消息到历史中
        messages.append({"role": "user", "content": _prompt})

        data = {
            "model": self.__model,
            "messages": messages
        }

        # 尝试进行对话链接
        try:
            response = requests.post(self.__url, json=data, headers=headers)
            # 检查是否有错误发生
            response.raise_for_status()
            response_data = response.json()
            self.__update_history(_prompt, response_data)
            # 返回 JSON 格式的响应
            return response_data
        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Connection Error: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"Other Error: {err}")

    def __update_history(self, user_input, response_data):
        """更新聊天历史记录，保持 10 条以内

        Args:
            user_input: 用户输入的内容
            response_data: chatgpt 返回的内容

        Returns:

        """
        # 保存用户输入和模型的回应到历史记录
        if user_input not in self.__history:
            self.__history.append(user_input)
            # 保持历史记录不超过10条
        if len(self.__history) > 10:
            self.__history.pop(0)
        if 'choices' in response_data and response_data['choices']:
            model_response = response_data['choices'][0]['message']['content']
            self.__history.append(model_response)
            if len(self.__history) > 10:
                self.__history.pop(0)


if __name__ == '__main__':
    import toml

    cfg = toml.load("./config.toml")
    chat_session = ChatGPTSession(cfg["openai"]["url"], cfg["openai"]["api_key"], cfg["openai"]["model"])

    for _ in range(10):
        prompt = input()
        result = chat_session.call_chatgpt(prompt)
        if result:
            for _msg_id in range(len(result["choices"])):
                print(result["choices"][_msg_id]["message"]["content"])
