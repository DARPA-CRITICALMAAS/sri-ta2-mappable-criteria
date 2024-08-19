# imports
import argparse  # for running script from command line
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key
import re  # for matching endpoint from request URL
import tiktoken  # for counting tokens
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import PartialState

requests_filepath = 'logs_debug_MVT_local/mineral_system/prompts/SIR10-5070A.jsonl'
with open(requests_filepath, 'r') as f:
    requests = [json.loads(l) for l in f.readlines()]

model_path = '/Data/LLM/Llama-2-13b-chat-hf'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=bnb_config, attn_implementation="flash_attention_2")
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, attn_implementation="flash_attention_2")
# model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, attn_implementation="flash_attention_2")
# model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)

def generate(model, tokenizer, messages, top_p=0.92, temperature=0.5, max_new_tokens=256):
    model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt").to("cuda")
    # print(tokenizer.batch_decode(model_inputs))
    input_length = model_inputs.shape[1]
    # TODO: temperature / top_p?
    generated_ids = model.generate(model_inputs, do_sample=True, top_p=top_p, max_new_tokens=max_new_tokens)
    response = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0]
    return response

distributed_state = PartialState()

model.to(distributed_state.device)

with distributed_state.split_between_processes(requests) as prompts:
    for prompt in prompts:
        result = generate(model, tokenizer, prompt['messages'])
        print(result)