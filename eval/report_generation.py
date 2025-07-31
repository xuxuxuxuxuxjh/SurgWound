import json
import random
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import base64
import io
from collections import defaultdict
import json
import re
import torch
import os
import pandas as pd
from peft import PeftModel





base_model_path = ''
# lora_model_path = ''
print("Loading base model...")
print(base_model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="auto",
)

# print("Loading LoRA adapter...")
# print(lora_model_path)
# model = PeftModel.from_pretrained(model, lora_model_path)

tokenizer = AutoProcessor.from_pretrained(
    base_model_path,
)

prompt='''
<image>\nBased on the image, generate a detailed medical report that includes the following aspects: wound location, wound status, closure method, exudate characteristics, presence of erythema, presence of edema, urgency level, infection risk assessment.
'''

def generate_image_caption(image_path, addition_text):
    
    image = Image.open(image_path).convert('RGB')
                    
    messages = [
        {
            "role": "system",
            "content": "You are a professional medical assessment expert specializing in wound evaluation and diagnosis. You are given a wound image and a question about the wound. You need to answer the question based on the image and your medical knowledge."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt + '\n' + addition_text}
            ]
        }
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = tokenizer(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response

save_file = '.xlsx'

# 检查xlsx文件是否存在，如果存在则读取现有数据
existing_data = {}
if os.path.exists(save_file):
    try:
        df = pd.read_excel(save_file)
        existing_data = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))  # 第一列是image_name，第二列是report
    except Exception as e:
        print(f"读取现有xlsx文件时出错: {e}")
        existing_data = {}

data_path = 'test_report.json'
data_list = []
try:
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            image_path = data['image_path']
            question = data['question']
            answer = data['answer']
            data_list.append(data)
except Exception as e:
    print(f"读取JSON文件时出错: {e}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)  # 直接加载整个JSON数组
    print(f"成功加载 {len(data_list)} 条数据")

# 存储新的结果
new_results = []

for data in data_list:
    image_name = data['image_name']
    image_path = os.path.join('', image_name)
    question = data['question']
    answer = data['answer']

    df = pd.read_excel('/records_processed.xlsx')
    row = df[df['Image Name'] == image_name].iloc[0]
    wound_location = row['Wound Location']
    
    df2 = pd.read_excel('.xlsx')
    row2 = df2[df2['Image Name'] == image_name].iloc[0]
    closure_method = row2['Closure Method'] 
    wound_status = row2['Wound Status']
    exudate_characteristics = row2['Exudate Characteristics']  
    erythema = row2['Erythema']
    edema = row2['Edema']
    urgency = row2['Urgency Level']
    risk = row2['Infection Risk Assessment']

    addition_text = ""
    addition_text += f" The wound location is {wound_location}."
    addition_text += f" The closure method is {closure_method}."
    addition_text += f" The wound status is {wound_status}."
    addition_text += f" The exudate characteristics is {exudate_characteristics}."
    addition_text += f" The erythema is {erythema}."
    addition_text += f" The edema is {edema}."
    addition_text += f" The urgency level is {urgency}."
    addition_text += f" The infection risk assessment is {risk}."
    
    # 检查是否已经处理过这个image_name
    if image_name in existing_data:
        print(f"跳过已存在的记录: {image_name}")
        continue
    
    print(f"处理图片: {image_name}")
    report = generate_image_caption(image_path, addition_text)

    print(report)
    
    # 添加到新结果列表
    new_results.append({
        'image_name': image_name,
        'report': report
    })

    

# 如果有新结果，保存到xlsx文件
if new_results:
    # 创建新的DataFrame
    new_df = pd.DataFrame(new_results)
    
    if os.path.exists(save_file):
        # 如果文件存在，追加新数据
        existing_df = pd.read_excel(save_file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # 如果文件不存在，创建新文件
        combined_df = new_df
    
    # 保存到xlsx文件
    combined_df.to_excel(save_file, index=False, columns=['image_name', 'report'])
    print(f"已保存 {len(new_results)} 条新记录到 {save_file}")
else:
    print("没有新的记录需要保存")
