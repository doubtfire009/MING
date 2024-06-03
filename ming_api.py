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
import numpy as np
import pdb
from transformers import TextStreamer

# from ming.serve.inference import chat_loop, ChatIO

class ChatInfo(BaseModel):
    query: str
    history: list
    temperature: float
    max_new_tokens: int

app = FastAPI()

class MingModel():
    def __init__(self, stream=True):
        # embedding_model_path = '/www/llm_model/X-D-Lab/MindChat-Qwen-7B-v2'
        model_path = '/www/llm_model/MING-7B'
        conv_template = 'bloom'
        device = 'GPU'
        model_base = None
        tokenizer, model, context_len, _ = load_pretrained_model(model_path, model_base, None, use_logit_bias=None, only_load=None)
        model.config.use_cache = True
        model.eval()
        # print("=== load llm ===")
        self.beam_size = 3
        self.tokenizer = tokenizer
        self.context_len = context_len
        #model = model.cuda()
        print("model_device:", next(model.parameters()).device) 
        self.model = model
        self.stream = True

    def launch_model_by_gpu(self):
        self.model = self.model.cuda()

    def create_chat_prompt(self, query, history):
        chat_prompt = ''
        history_prompt = ''
        if len(history) > 0:
            for item in history:
                history_prompt = ' ' + chat_prompt + 'USER: ' + item[0] + ' ASSISTANT: ' + item[1]

        chat_prompt = history_prompt + ' USER: ' + query + ' ASSISTANT:'
        return chat_prompt

    @torch.inference_mode()
    def create_history_chat(self, query, history, temperature, max_new_tokens):
        chat_prompt = self.create_chat_prompt(query, history)
        prompt = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""
        prompt = prompt + ' ' + chat_prompt
        stop_str = '</s>'
        if stop_str == self.tokenizer.eos_token:
            stop_str = None
        params = {
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": stop_str,
        }
        input_ids = self.tokenizer(prompt).input_ids
        # output_ids = list(input_ids)

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = torch.tensor(input_ids[-max_src_len:]).unsqueeze(0).cuda()
        print("model type:", type(self.model))
        print("input_ids:", type(input_ids))
        print("max_new_tokens:", type(max_new_tokens))
        print("beam_size:", type(self.beam_size))
        print("temperature:", type(temperature))
        print("model_device--gen:", next(self.model.parameters()).device)
        
        current_gpu_index = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory / (1024 ** 3)  # 显存总量(GB)
        used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
        free_memory = total_memory - used_memory
        print("free_memory:", free_memory)
        print("used_memory:", used_memory)

        if self.stream:
            streamer = TextStreamer(self.tokenizer)
            outputs = self.model.generate(
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=self.beam_size,
                temperature=temperature,
                streamer=streamer

            )
            outputs = outputs[0][len(input_ids[0]):]
            output = self.tokenizer.decode(outputs, skip_special_tokens=True)
        else:
            outputs = self.model.generate(
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_beams=self.beam_size,
                temperature=temperature,
            )
            outputs = outputs[0][len(input_ids[0]):]
            output = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return output

ming_model = MingModel()
ming_model.launch_model_by_gpu()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/doctor/consult")
def doctor_consult(chatInfo: ChatInfo):
    query, history, temperature, max_new_tokens = chatInfo.query, chatInfo.history, chatInfo.temperature, chatInfo.max_new_tokens
    response = ming_model.create_history_chat(query, history, temperature, max_new_tokens)
    return response

if __name__ == '__main__':

    uvicorn.run(app='ming_api:app', host="0.0.0.0", port=8081, reload=True)
