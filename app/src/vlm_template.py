import os
import sys
import torch
from openai import OpenAI
from transformers import (
    LlavaNextProcessor, LlavaNextForConditionalGeneration, 
    Qwen2VLForConditionalGeneration, Qwen2VLProcessor
)
## init device
device = "cpu"
torch_dtype = torch.float16


vlms_list = [
    # {
    #     "type": "llava-next",
    #     "name": "llava-v1.6-mistral-7b-hf",
    #     "local_path": "models/vlms/llava-v1.6-mistral-7b-hf",
    #     "processor": LlavaNextProcessor.from_pretrained(
    #         "models/vlms/llava-v1.6-mistral-7b-hf"
    #     ) if os.path.exists("models/vlms/llava-v1.6-mistral-7b-hf") else LlavaNextProcessor.from_pretrained(
    #         "llava-hf/llava-v1.6-mistral-7b-hf"
    #     ),
    #     "model": LlavaNextForConditionalGeneration.from_pretrained(
    #         "models/vlms/llava-v1.6-mistral-7b-hf", torch_dtype=torch_dtype, device_map=device
    #     ).to("cpu") if os.path.exists("models/vlms/llava-v1.6-mistral-7b-hf") else 
    #         LlavaNextForConditionalGeneration.from_pretrained(
    #             "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch_dtype, device_map=device
    #         ).to("cpu"),
    # },
    # {
    #     "type": "llava-next",
    #     "name": "llama3-llava-next-8b-hf (Preload)",
    #     "local_path": "models/vlms/llama3-llava-next-8b-hf",
    #     "processor": LlavaNextProcessor.from_pretrained(
    #         "models/vlms/llama3-llava-next-8b-hf"
    #     ) if os.path.exists("models/vlms/llama3-llava-next-8b-hf") else LlavaNextProcessor.from_pretrained(
    #         "llava-hf/llama3-llava-next-8b-hf"
    #     ),
    #     "model": LlavaNextForConditionalGeneration.from_pretrained(
    #         "models/vlms/llama3-llava-next-8b-hf", torch_dtype=torch_dtype, device_map=device
    #     ).to("cpu") if os.path.exists("models/vlms/llama3-llava-next-8b-hf") else 
    #         LlavaNextForConditionalGeneration.from_pretrained(
    #             "llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch_dtype, device_map=device
    #         ).to("cpu"),
    # },
    # {
    #     "type": "llava-next",
    #     "name": "llava-v1.6-vicuna-13b-hf",
    #     "local_path": "models/vlms/llava-v1.6-vicuna-13b-hf",
    #     "processor": LlavaNextProcessor.from_pretrained(
    #         "models/vlms/llava-v1.6-vicuna-13b-hf"
    #     ) if os.path.exists("models/vlms/llava-v1.6-vicuna-13b-hf") else LlavaNextProcessor.from_pretrained(
    #         "llava-hf/llava-v1.6-vicuna-13b-hf"
    #     ),
    #     "model": LlavaNextForConditionalGeneration.from_pretrained(
    #         "models/vlms/llava-v1.6-vicuna-13b-hf", torch_dtype=torch_dtype, device_map=device
    #     ).to("cpu") if os.path.exists("models/vlms/llava-v1.6-vicuna-13b-hf") else 
    #         LlavaNextForConditionalGeneration.from_pretrained(
    #             "llava-hf/llava-v1.6-vicuna-13b-hf", torch_dtype=torch_dtype, device_map=device
    #         ).to("cpu"),
    # },
    # {
    #     "type": "llava-next",
    #     "name": "llava-v1.6-34b-hf",
    #     "local_path": "models/vlms/llava-v1.6-34b-hf",
    #     "processor": LlavaNextProcessor.from_pretrained(
    #         "models/vlms/llava-v1.6-34b-hf"
    #     ) if os.path.exists("models/vlms/llava-v1.6-34b-hf") else LlavaNextProcessor.from_pretrained(
    #         "llava-hf/llava-v1.6-34b-hf"
    #     ),
    #     "model": LlavaNextForConditionalGeneration.from_pretrained(
    #         "models/vlms/llava-v1.6-34b-hf", torch_dtype=torch_dtype, device_map=device
    #     ).to("cpu") if os.path.exists("models/vlms/llava-v1.6-34b-hf") else 
    #         LlavaNextForConditionalGeneration.from_pretrained(
    #             "llava-hf/llava-v1.6-34b-hf", torch_dtype=torch_dtype, device_map=device
    #         ).to("cpu"),
    # },
    # {
    #     "type": "qwen2-vl",
    #     "name": "Qwen2-VL-2B-Instruct",
    #     "local_path": "models/vlms/Qwen2-VL-2B-Instruct",
    #     "processor": Qwen2VLProcessor.from_pretrained(
    #         "models/vlms/Qwen2-VL-2B-Instruct"
    #     ) if os.path.exists("models/vlms/Qwen2-VL-2B-Instruct") else Qwen2VLProcessor.from_pretrained(
    #         "Qwen/Qwen2-VL-2B-Instruct"
    #     ),
    #     "model": Qwen2VLForConditionalGeneration.from_pretrained(
    #         "models/vlms/Qwen2-VL-2B-Instruct", torch_dtype=torch_dtype, device_map=device
    #     ).to("cpu") if os.path.exists("models/vlms/Qwen2-VL-2B-Instruct") else 
    #         Qwen2VLForConditionalGeneration.from_pretrained(
    #             "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch_dtype, device_map=device
    #         ).to("cpu"),
    # },
    # {
    #     "type": "qwen2-vl",
    #     "name": "Qwen2-VL-7B-Instruct (Default)",
    #     "local_path": "models/vlms/Qwen2-VL-7B-Instruct",
    #     "processor": Qwen2VLProcessor.from_pretrained(
    #         "models/vlms/Qwen2-VL-7B-Instruct"
    #     ) if os.path.exists("models/vlms/Qwen2-VL-7B-Instruct") else Qwen2VLProcessor.from_pretrained(
    #         "Qwen/Qwen2-VL-7B-Instruct"
    #     ),
    #     "model": Qwen2VLForConditionalGeneration.from_pretrained(
    #         "models/vlms/Qwen2-VL-7B-Instruct", torch_dtype=torch_dtype, device_map=device
    #     ).to("cpu") if os.path.exists("models/vlms/Qwen2-VL-7B-Instruct") else 
    #         Qwen2VLForConditionalGeneration.from_pretrained(
    #             "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch_dtype, device_map=device
    #         ).to("cpu"),
    # },
    {
        "type": "openai",
        "name": "GPT4-o (Highly Recommended)",
        "local_path": "",
        "processor": "",
        "model": ""
    },
]

vlms_template = {k["name"]: (k["type"], k["local_path"], k["processor"], k["model"]) for k in vlms_list}