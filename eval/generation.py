import json
import pandas as pd
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import re

OPTIONS = {
    'Wound Location': ['Abdomen', 'Patella', 'Ankle', 'Facial region', 'Manus', 'Cervical region', 'Other'],
    'Wound Status': ['Healed', 'Not Healed'],
    'Closure Method': ['Invisible', 'Sutures', 'Staples', 'Adhesives'],
    'Exudate Characteristics': ['Non-existent', 'Serous', 'Sanguineous', 'Purulent', 'Seropurulent'],
    'Erythema': ['Non-existent', 'Existent'],
    'Edema': ['Non-existent', 'Existent'],
    'Urgency Level': [
        'Home Care (Green): Manage with routine care',
        'Clinic Visit (Yellow): Requires professional evaluation within 48 hours',
        'Emergency Care (Red): Seek immediate medical attention'
    ],
    'Infection Risk Assessment': ['Low', 'Medium', 'High']
}

def format_question_with_options_for_eval(question, options):
    """格式化问题和选项"""
    formatted_question = question + "\n\n"
    for i, option in enumerate(options):
        formatted_question += f"{chr(65+i)}. {option}\n"
    formatted_question += "\nPlease answer with the only one letter (A, B, C, D, etc.):"
    return formatted_question

def load_model(model_path):
    """加载模型"""
    print(f"Loading model from {model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoProcessor.from_pretrained(model_path)
    return model, tokenizer

def predict_single_field(image_path, field_name, model, tokenizer):
    """对单个字段进行预测"""
    try:
        with torch.no_grad():
            image = Image.open(image_path).convert('RGB')
            
            # 构造问题
            if field_name == 'Wound Status':
                question = "What is the current healing status of this surgical wound?"
            elif field_name == 'Closure Method':
                question = "What closure method was used for this surgical wound?"
            elif field_name == 'Exudate Characteristics':
                question = "What type of exudate is present in this surgical wound?"
            elif field_name == 'Erythema':
                question = "Is there erythema (redness) present around this surgical wound?"
            elif field_name == 'Edema':
                question = "Has edema developed in this surgical wound?"
            else:
                return "Unknown"
            
            options = OPTIONS[field_name]
            formatted_question = format_question_with_options_for_eval(question, options)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": formatted_question}
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

            # 推理
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            # 提取答案
            match = re.search(r'[A-Z]', response)
            if match:
                predicted_answer = match.group()
                answer_index = ord(predicted_answer) - ord('A')
                if 0 <= answer_index < len(options):
                    return options[answer_index]
            
            return "Unknown"
            
    except Exception as e:
        print(f"Error predicting {field_name} for {image_path}: {e}")
        return "Error"

# 读取数据
path = ''

data_list = []
with open(path, 'r') as f:
    for line in f:
        data = json.loads(line)
        data_list.append(data)

# 分类数据
location_list = []
status_list = []
closure_list = []
exudate_list = []
erythema_list = []
edema_list = []
urgency_list = []
risk_list = []

for data in data_list:
    if data['field'] == 'Wound Location':
        location_list.append(data)
    elif data['field'] == 'Wound Status':
        status_list.append(data)
    elif data['field'] == 'Closure Method':
        closure_list.append(data)
    elif data['field'] == 'Exudate Characteristics':
        exudate_list.append(data)
    elif data['field'] == 'Erythema':
        erythema_list.append(data)
    elif data['field'] == 'Edema':
        edema_list.append(data)
    elif data['field'] == 'Urgency Level':
        urgency_list.append(data)
    elif data['field'] == 'Infection Risk Assessment':
        risk_list.append(data)

# 获取risk_list中所有独特的图片路径
risk_images = set()
for data in risk_list:
    risk_images.add(data['image_name'])

print(f"Found {len(risk_images)} unique images in risk_list")

# 模型路径配置 - 需要根据你的实际路径修改
model_paths = {
    'Wound Status': '',
    'Closure Method': '', 
    'Exudate Characteristics': '',
    'Erythema': '',
    'Edema': ''
}


# 初始化结果字典，每张图片一个条目
results = {}
for image_name in risk_images:
    results[image_name] = {
        'Image_Name': image_name,
        'Doctor_Name': 'Doctor_B',  # 可根据需要修改
        'Age': 'Normal',  # 可根据需要修改  
        'Wound_Location': '',  # 需要从其他数据获取或预测
        'Wound_Status': '',
        'Closure_Method': '',
        'Exudate_Characteristics': '',
        'Erythema': '',
        'Edema': '',
        'Urgency_Level': '',  # 需要从其他数据获取或预测
        'Risk_Level': ''  # 需要从其他数据获取或预测
    }


# 对每个字段分别加载模型并进行推理
for field_name in ['Wound Status', 'Closure Method', 'Exudate Characteristics', 'Erythema', 'Edema']:
    print(f"\n处理字段: {field_name}")
    
    # 加载对应的模型
    model, tokenizer = load_model(model_paths[field_name])
    
    # 对每张图片进行推理
    for i, image_name in enumerate(risk_images):
        image_path = ''
        print(f"Processing image {i+1}/{len(risk_images)}: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            prediction = "Image Not Found"
        else:
            prediction = predict_single_field(image_path, field_name, model, tokenizer)
        
        # 根据字段名映射到正确的列名
        field_mapping = {
            'Wound Status': 'Wound_Status',
            'Closure Method': 'Closure_Method', 
            'Exudate Characteristics': 'Exudate_Characteristics',
            'Erythema': 'Erythema',
            'Edema': 'Edema'
        }
        
        results[image_name][field_mapping[field_name]] = prediction
    
    # 清理GPU内存
    del model, tokenizer
    torch.cuda.empty_cache()

# 转换为DataFrame并保存
results_list = list(results.values())
df = pd.DataFrame(results_list)

# 重命名列以匹配你的Excel格式
df.columns = ['Image_Name', 'Doctor_Name', 'Age', 'Wound_Location', 'Wound_Status', 
              'Closure_Method', 'Exudate_Characteristics', 'Erythema', 'Edema', 
              'Urgency_Level', 'Risk_Level']

df.to_excel('result_train.xlsx', index=False)
print(f"\n结果已保存到 result_train.xlsx，共 {len(results_list)} 条记录")
