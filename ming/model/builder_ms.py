import os
import warnings
import shutil
from copy import copy
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers import BitsAndBytesConfig
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
from ming.model.utils import get_mixoflora_model
import json


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, use_logit_bias=False,
                          only_load=None, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if model_base is not None:
        # PEFT model
        from peft import PeftModel, PeftConfig
        tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_base, trust_remote_code=True, low_cpu_mem_usage=True,
                                                     **kwargs)
        print(f"Loading LoRA weights from {model_path}")
        lora_config = PeftConfig.from_pretrained(model_path)
        if only_load == "attn":
            lora_config.target_modules = {m for m in lora_config.target_modules if
                                          m not in ["up_proj", "down_proj", "gate_proj"]}

        elif only_load == "ffn":
            lora_config.target_modules = {m for m in lora_config.target_modules if
                                          m in ["up_proj", "down_proj", "gate_proj"]}

        model = PeftModel.from_pretrained(model, model_path, config=lora_config)
        print(f"Merging weights")
        model = model.merge_and_unload()
        print('Convert to FP16...')
        model.to(torch.float16)

        tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_base, add_prefix_space=True,
                                                                    trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True,
                                                     **kwargs)

        tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(model_path, add_prefix_space=True,
                                                                    trust_remote_code=True)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len, tokenizer_with_prefix_space
