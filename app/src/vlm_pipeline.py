import base64
import re
import torch

from PIL import Image
from io import BytesIO
import numpy as np
import gradio as gr

from openai import OpenAI
from transformers import (LlavaNextForConditionalGeneration, Qwen2VLForConditionalGeneration)
from qwen_vl_utils import process_vision_info

from app.gpt4_o.instructions import (
    create_editing_category_messages_gpt4o, 
    create_ori_object_messages_gpt4o, 
    create_add_object_messages_gpt4o,
    create_apply_editing_messages_gpt4o)

from app.llava.instructions import (
    create_editing_category_messages_llava, 
    create_ori_object_messages_llava, 
    create_add_object_messages_llava,
    create_apply_editing_messages_llava)

from app.qwen2.instructions import (
    create_editing_category_messages_qwen2, 
    create_ori_object_messages_qwen2, 
    create_add_object_messages_qwen2,
    create_apply_editing_messages_qwen2)

from app.utils.utils import run_grounded_sam


def encode_image(img):
    img = Image.fromarray(img.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def run_gpt4o_vl_inference(vlm_model, 
                           messages):
    response = vlm_model.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages
    )
    response_str = response.choices[0].message.content
    return response_str

def run_llava_next_inference(vlm_processor, vlm_model, messages, image, device="cuda"):
    prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = vlm_processor(images=image, text=prompt, return_tensors="pt").to(device)
    output = vlm_model.generate(**inputs, max_new_tokens=200)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output)
    ]
    response_str = vlm_processor.decode(generated_ids_trimmed[0], skip_special_tokens=True)
   
    return response_str

def run_qwen2_vl_inference(vlm_processor, vlm_model, messages, image, device="cuda"):
    text = vlm_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    generated_ids = vlm_model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response_str = vlm_processor.decode(generated_ids_trimmed[0], skip_special_tokens=True)
    return response_str


### response editing type
def vlm_response_editing_type(vlm_processor, 
                              vlm_model, 
                              image, 
                              editing_prompt,
                              device):

    if isinstance(vlm_model, OpenAI):
        messages = create_editing_category_messages_gpt4o(editing_prompt)
        response_str = run_gpt4o_vl_inference(vlm_model, messages)
    elif isinstance(vlm_model, LlavaNextForConditionalGeneration):
        messages = create_editing_category_messages_llava(editing_prompt)
        response_str = run_llava_next_inference(vlm_processor, vlm_model, messages, image, device=device)
    elif isinstance(vlm_model, Qwen2VLForConditionalGeneration):
        messages = create_editing_category_messages_qwen2(editing_prompt)
        response_str = run_qwen2_vl_inference(vlm_processor, vlm_model, messages, image, device=device)
    
    try:
        for category_name in ["Addition","Remove","Local","Global","Background"]:
            if category_name.lower() in response_str.lower():
                return category_name
    except Exception as e:
        raise gr.Error("Please input OpenAI API Key. Or please input correct commands, including add, delete, and modify commands. If it still does not work, please switch to a more powerful VLM.")


### response object to be edited        
def vlm_response_object_wait_for_edit(vlm_processor, 
                                      vlm_model, 
                                      image, 
                                      category, 
                                      editing_prompt,
                                      device):
    if category in ["Background", "Global", "Addition"]:
        edit_object = "nan"
        return edit_object

    if isinstance(vlm_model, OpenAI):
        messages = create_ori_object_messages_gpt4o(editing_prompt)
        response_str = run_gpt4o_vl_inference(vlm_model, messages)
    elif isinstance(vlm_model, LlavaNextForConditionalGeneration):
        messages = create_ori_object_messages_llava(editing_prompt)
        response_str = run_llava_next_inference(vlm_processor, vlm_model, messages, image , device)
    elif isinstance(vlm_model, Qwen2VLForConditionalGeneration):
        messages = create_ori_object_messages_qwen2(editing_prompt)
        response_str = run_qwen2_vl_inference(vlm_processor, vlm_model, messages, image, device)
    return response_str


### response mask
def vlm_response_mask(vlm_processor, 
                      vlm_model, 
                      category, 
                      image, 
                      editing_prompt, 
                      object_wait_for_edit, 
                      sam=None,
                      sam_predictor=None,
                      sam_automask_generator=None,
                      groundingdino_model=None,
                      device=None,
                      ):
    mask = None
    if editing_prompt is None or len(editing_prompt)==0:
        raise gr.Error("Please input the editing instruction!")
    height, width = image.shape[:2]
    if category=="Addition":
        try:
            if isinstance(vlm_model, OpenAI):
                base64_image = encode_image(image)
                messages = create_add_object_messages_gpt4o(editing_prompt, base64_image, height=height, width=width)
                response_str = run_gpt4o_vl_inference(vlm_model, messages)
            elif isinstance(vlm_model, LlavaNextForConditionalGeneration):
                messages = create_add_object_messages_llava(editing_prompt, height=height, width=width)
                response_str = run_llava_next_inference(vlm_processor, vlm_model, messages, image, device)
            elif isinstance(vlm_model, Qwen2VLForConditionalGeneration):
                base64_image = encode_image(image)
                messages = create_add_object_messages_qwen2(editing_prompt, base64_image, height=height, width=width)
                response_str = run_qwen2_vl_inference(vlm_processor, vlm_model, messages, image, device)
            print(response_str)
            pattern = r'\[\d{1,3}(?:,\s*\d{1,3}){3}\]'
            box = re.findall(pattern, response_str)
            box = box[0][1:-1].split(",")
            for i in range(len(box)):
                box[i] = int(box[i])
            cus_mask = np.zeros((height, width))
            cus_mask[box[1]: box[1]+box[3], box[0]: box[0]+box[2]]=255
            mask = cus_mask
        except:
            raise gr.Error("Please set the mask manually, currently the VLM cannot output the mask!")

    elif category=="Background":
        labels = "background"
    elif category=="Global":
        mask = 255 * np.zeros((height, width))
    else:
        labels = object_wait_for_edit
    
    if mask is None:
        for thresh in [0.3,0.25,0.2,0.15,0.1,0.05,0]:
            try:
                detections = run_grounded_sam(
                    input_image={"image":Image.fromarray(image.astype('uint8')),
                                 "mask":None}, 
                    text_prompt=labels, 
                    task_type="seg", 
                    box_threshold=thresh, 
                    text_threshold=0.25, 
                    iou_threshold=0.5, 
                    scribble_mode="split",
                    sam=sam,
                    sam_predictor=sam_predictor,
                    sam_automask_generator=sam_automask_generator,
                    groundingdino_model=groundingdino_model,
                    device=device,
                )
                mask = np.array(detections[0,0,...].cpu()) * 255
                break
            except:
                print(f"wrong in threshhold: {thresh}, continue")
                continue
    return mask


def vlm_response_prompt_after_apply_instruction(vlm_processor, 
                                                vlm_model, 
                                                image, 
                                                editing_prompt,
                                                device):
                                                
    try:
        if isinstance(vlm_model, OpenAI):
            base64_image = encode_image(image)  
            messages = create_apply_editing_messages_gpt4o(editing_prompt, base64_image)
            response_str = run_gpt4o_vl_inference(vlm_model, messages)
        elif isinstance(vlm_model, LlavaNextForConditionalGeneration):
            messages = create_apply_editing_messages_llava(editing_prompt)
            response_str = run_llava_next_inference(vlm_processor, vlm_model, messages, image, device)
        elif isinstance(vlm_model, Qwen2VLForConditionalGeneration):
            base64_image = encode_image(image)  
            messages = create_apply_editing_messages_qwen2(editing_prompt, base64_image)
            response_str = run_qwen2_vl_inference(vlm_processor, vlm_model, messages, image, device)
        else:
            raise gr.Error("Please select the correct VLM model and input the correct API Key first!")
    except Exception as e:
        raise gr.Error("Please select the correct VLM model and input the correct API Key first!")
    return response_str