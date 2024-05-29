import gradio as gr
from vllm import LLM, SamplingParams
# from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os
import abc
from typing import Optional
import warnings
import time
import torch
from ming.conversations import conv_templates, get_default_conv_template, SeparatorStyle
from ming.model.builder import load_pretrained_model, load_molora_pretrained_model
import numpy as np
import pdb
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import abc
from typing import Optional
import warnings
import time
import torch
from ming.conversations import conv_templates, get_default_conv_template, SeparatorStyle
from ming.model.builder import load_pretrained_model, load_molora_pretrained_model


os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ['VLLM_USE_MODELSCOPE']='True'

# class ming_llm():
#     def __init__(self):
#         # embedding_model_path = '/www/llm_model/X-D-Lab/MindChat-Qwen-7B-v2'
#         llm_model_path = '/www/llm_model/MING-7B'
#         device = 'GPU'
#         print("=== load llm ===")
#         # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
#         # llm = LLM(model=llm_model_path, trust_remote_code=True, gpu_memory_utilization=0.9)
#         self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map="auto", trust_remote_code=True, fp16=True).eval()
#         model.generation_config = GenerationConfig.from_pretrained(llm_model_path, revision='v1.0.1', trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参
#         self.mindchat_model = model
#     def create_history_chat(self, question, bot):
#         chat_history = []
#         for item in bot:
#             chat_history.append((item[0], item[1]))
#
#         response, history = self.mindchat_model.chat(self.tokenizer, question, history=None)
#         print("------test------")
#         print(response)
#         print("-----history-------")
#         print(history)
#         return response
#
# mt_llm = mindchat_llm()

import json
import requests


def request_post(url, param):
    fails = 0
    text = ''
    while True:
        try:
            if fails >= 20:
                break

            headers = {'content-type': 'application/json'}
            ret = requests.post(url, json=param, headers=headers, timeout=10)

            if ret.status_code == 200:
                text = json.loads(ret.text)
            else:
                continue
        except:
            fails += 1
            print('网络连接出现问题, 正在尝试再次请求: ', fails)
        else:
            break
    return text


def doChatbot(message, bot):
    post_url = "http://localhost:8081/doctor/consult"

    chat_history = []
    for item in bot:
        chat_history.append([item[0], item[1]])

    request_param = {"query": message, "history": chat_history, "temperature": 1.2, "max_new_tokens": 2048}

    response = request_post(post_url, request_param)
    # if "我: " in answer:
    #     res = answer.split("我: ")[-1]
    # elif "AI:" in answer:
    #     res = answer.split("AI:")[-1]
    # elif "？" in answer:
    #     res = answer.split("？")[-1]
    # elif "?" in answer:
    #     res = answer.split("?")[-1]
    # else:
    #     res = answer
    return response

def start_chatbot():
    gr.ChatInterface(
        fn=doChatbot,
        chatbot=gr.Chatbot(height=500, value=[]),
        textbox=gr.Textbox(placeholder="请输入您的问题", container=False, scale=7),
        title="老白医学咨询（MING-7b）",
        theme="soft",
        submit_btn="发送",
        clear_btn="清空"
    ).queue().launch(server_port=7001, server_name='0.0.0.0')

# if __name__ == "__main__":


start_chatbot()