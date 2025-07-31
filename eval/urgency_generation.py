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



TARGET_ATTRS = [
    'Urgency Level',
]

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



path = 'test_question.json'

data_list = []
with open(path, 'r') as f:
    for line in f:
        data = json.loads(line)
        data_list.append(data)

# print(len(data_list))
# print(data_list[0])

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

# print(len(location_list))
# print(len(status_list))
# print(len(closure_list))
# print(len(exudate_list))
# print(len(erythema_list))
# print(len(edema_list))
# print(len(urgency_list))

def format_question_with_options(question, options):
    """格式化问题和选项"""
    if question[0] == 'W':
        question = 'w' + question[1:]
    question = '<image>\nBased on the image, ' + question
    formatted_question = question + "\nOptions:\n"
    for i, option in enumerate(options):
        formatted_question += f"{chr(65+i)}. {option}\n"
    formatted_question += f"Please answer with the only one choice ({', '.join([chr(ord('A') + i) + '. ' + opt for i, opt in enumerate(options)])}):"
    return formatted_question

def format_question_with_options_for_eval(question, options):
    """格式化问题和选项用于评估"""
    if question[0] == 'W':
        question = 'w' + question[1:]
    question = '<image>\nBased on the image, ' + question
    formatted_question = question + "\nOptions:\n"
    for i, option in enumerate(options):
        formatted_question += f"{chr(65+i)}. {option}\n"
    formatted_question += "Please answer with the only one letter (A, B, C, D, etc.):"
    return formatted_question

urgency_processed_list = []
image_path = ''
for data in urgency_list:
    question = data['question']
    options = data['options']
    random.shuffle(options)
    answer = data['answer']

    question_text = format_question_with_options(question, options)

    # 找到答案对应的字母
    answer_index = data["options"].index(data["answer"])
    answer_letter = chr(ord('A') + answer_index)
    answer_text = f"{answer_letter}. {data['answer']}"

    urgency_processed_list.append(
        {
            "messages": [
                {
                    "content": question_text,
                    "role": "user"
                },
                {
                    "content": answer_text,
                    "role": "assistant"
                }
            ],
            "images": [
                image_path + data["image_name"]
            ]
        }
    )

print(len(urgency_processed_list))


test_path = 'test_question.json'
test_question_dataset = []
with open(test_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        if data['field'] == 'Urgency Level':
            test_question_dataset.append(data)

print(len(test_question_dataset))


base_model_path = ''
lora_model_path = ''
print("Loading base model...")
print(base_model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="auto",
)

print("Loading LoRA adapter...")
print(lora_model_path)
model = PeftModel.from_pretrained(model, lora_model_path)

tokenizer = AutoProcessor.from_pretrained(
    base_model_path,
)

def decode_base64_image(base64_string):
    """将base64编码的图像解码为PIL Image"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def format_question_with_options(question, options):
    """格式化问题和选项"""
    formatted_question = question + "\n\n"
    for i, option in enumerate(options):
        formatted_question += f"{chr(65+i)}. {option}\n"
    formatted_question += "\nPlease answer with the only one letter (A, B, C, D, etc.):"
    return formatted_question

def evaluate_model(dataset, model, tokenizer):
    """评估模型性能"""
    results = []
    field_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    TP = defaultdict(lambda: defaultdict(int))
    FP = defaultdict(lambda: defaultdict(int))
    FN = defaultdict(lambda: defaultdict(int))
    
    print(f"Starting evaluation on {len(dataset)} samples...")
    
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i % 100 == 0:
                print(f"Processing sample {i}/{len(dataset)}")
            
            try:
                # 获取样本信息
                sample_id = sample['id']
                field = sample['field']
                question = sample['question']
                question = f"What is the urgency level of this surgical wound?"
                df = pd.read_excel('records_processed.xlsx')
                image_name = sample['image_name'].split('.')[0].split('_')[0] + '.jpg'
                row = df[df['Image Name'] == image_name].iloc[0]
                df2 = pd.read_excel('result.xlsx')
                row2 = df2[df2['Image_Name'] == image_name].iloc[0]
                wound_location = row['Wound Location']
                closure_method = row2['Closure_Method']
                wound_status = row2['Wound_Status']
                exudate_characteristics = row2['Exudate_Characteristics']
                erythema = row2['Erythema']
                edema = row2['Edema']

                question = data['question']
                question = f"what is the urgency level of this surgical wound?"
                if wound_location not in ['Uncertain', 'Other']:
                    question += f" The wound location is {wound_location}."
                if closure_method not in ['Uncertain']:
                    question += f" The closure method is {closure_method}."
                if wound_status not in ['Uncertain']:
                    question += f" The wound status is {wound_status}."
                if exudate_characteristics not in ['Uncertain']:
                    question += f" The exudate characteristics is {exudate_characteristics}."
                if erythema not in ['Uncertain']:
                    question += f" The erythema is {erythema}."
                if edema not in ['Uncertain']:
                    question += f" The edema is {edema}."
                options = sample['options']
                random.shuffle(options)
                correct_answer = sample['answer']
                image_name = sample['image_name']
                image_path = os.path.join('/fs/ess/PCON0023/xjh/data/wound_last_image', image_name)
                image = Image.open(image_path).convert('RGB')
                                
                # 格式化输入
                formatted_question = format_question_with_options_for_eval(question, options)
                if i == 0:
                    print(f'formatted_question: {formatted_question}')
                
                # 构建消息
                messages = [
                    {
                        "role": "system",
                        "content": "You are a professional medical assessment expert specializing in wound evaluation and diagnosis. You are given a wound image and a question about the wound. You need to answer the question based on the image and your medical knowledge."
                    },
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

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = tokenizer.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                print(f'response: {response}')
                
                # 提取答案，用正则表达式提取
                predicted_answer = re.search(r'[A-Z]', response).group() if re.search(r'[A-Z]', response) else None
                print(f'predicted_answer: {predicted_answer}')
                
                # 判断正确性， correct_answer 是 options 中的一个，需要转为A,B,C,D,E,F
                correct_answer = chr(65 + options.index(correct_answer))
                is_correct = predicted_answer == correct_answer
                # print(f'predict_answer: {predicted_answer}')
                print(f'correct_answer: {correct_answer}')
                print(f'is_correct: {is_correct}')
                print("")
                
                # 记录结果
                result = {
                    'id': sample_id,
                    'field': field,
                    'question': question,
                    'correct_answer': correct_answer + '. ' + options[ord(correct_answer)-65],
                    'predicted_answer': predicted_answer + '. ' + options[ord(predicted_answer)-65],
                    'is_correct': is_correct,
                    'response': response
                }
                results.append(result)
                
                # 更新统计
                field_stats[field]['total'] += 1
                if is_correct:
                    field_stats[field]['correct'] += 1
                    TP[field][options[ord(predicted_answer)-65]] += 1
                else:
                    FP[field][options[ord(predicted_answer)-65]] += 1
                    FN[field][options[ord(correct_answer)-65]] += 1
                    print(f'Image Name: {image_name} Correct Answer: {correct_answer} Predicted Answer: {predicted_answer}')
                    
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                continue
    
    return results, field_stats, TP, FP, FN

def print_statistics(field_stats, results, TP, FP, FN):
    """打印统计结果"""
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    total_correct = sum(stats['correct'] for stats in field_stats.values())
    total_samples = sum(stats['total'] for stats in field_stats.values())
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f"Overall Accuracy: {total_correct}/{total_samples} = {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print("\nAccuracy by Field:")
    print("-" * 40)
    
    for field, stats in sorted(field_stats.items()):
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{field:20s}: {stats['correct']:3d}/{stats['total']:3d} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    for attr in TARGET_ATTRS:
        print(f"\n{attr}:")
        # 收集所有选项
        all_options = set()
        all_options.update(TP[attr].keys())
        all_options.update(FP[attr].keys()) 
        all_options.update(FN[attr].keys())
        
        for option in sorted(all_options):
            if option and option != 'Uncertain':
                tp = TP[attr][option]
                fp = FP[attr][option]
                fn = FN[attr][option]
                
                # 计算precision, recall, f1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                print(f"  {option}: TP={tp}, FP={fp}, FN={fn}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

def update_excel_with_predictions(results, excel_path):
    """更新Excel文件中的Urgency_Level列"""
    try:
        # 读取现有的Excel文件
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
            print(f"Loaded existing Excel file with {len(df)} rows")
            print(f"Columns in Excel: {df.columns.tolist()}")
        else:
            print(f"Excel file {excel_path} not found, creating new one...")
            return
        
        # 创建一个字典来存储图片名称和预测结果的映射
        predictions_dict = {}
        for result in results:
            if 'predicted_answer' in result and result['predicted_answer']:
                # 从预测答案中提取实际的风险等级（去掉字母前缀）
                predicted_text = result['predicted_answer']
                if '. ' in predicted_text:
                    urgency_level = predicted_text.split('. ')[1]
                else:
                    urgency_level = predicted_text
                
                # 需要从test_question_dataset中获取对应的image_name
                sample_id = result.get('id', '')
                for sample in test_question_dataset:
                    if sample['id'] == sample_id:
                        image_name = sample['image_name']
                        # 转换图片名称格式以匹配Excel中的格式
                        base_name = image_name.split('.')[0].split('_')[0] + '.jpg'
                        predictions_dict[base_name] = urgency_level
                        print(f"Added prediction for {base_name}: {urgency_level}")
                        break
        
        print(f"Generated predictions for {len(predictions_dict)} images")
        
        # 尝试不同的可能列名
        possible_image_cols = ['Image_Name']
        possible_urgency_cols = ['Urgency_Level']
        
        image_col = None
        urgency_col = None
        
        # 找到正确的列名
        for col in possible_image_cols:
            if col in df.columns:
                image_col = col
                break
        
        for col in possible_urgency_cols:
            if col in df.columns:
                urgency_col = col
                break
        
        if image_col is None:
            print(f"Could not find image column. Available columns: {df.columns.tolist()}")
            return
        
        if urgency_col is None:
            print(f"Could not find urgency column. Available columns: {df.columns.tolist()}")
            return
        
        print(f"Using image column: {image_col}, urgency column: {urgency_col}")
        
        # 更新DataFrame中的Urgency_Level列
        updated_count = 0
        for idx, row in df.iterrows():
            image_name = row.get(image_col, '')
            if image_name in predictions_dict:
                df.at[idx, urgency_col] = predictions_dict[image_name]
                updated_count += 1
                print(f"Updated {image_name}: {predictions_dict[image_name]}")
        
        print(f"Updated {updated_count} rows in the Excel file")
        
        # 保存更新后的Excel文件
        df.to_excel(excel_path, index=False)
        print(f"Updated Excel file saved to {excel_path}")
        
    except Exception as e:
        print(f"Error updating Excel file: {e}")
        import traceback
        traceback.print_exc()

# 运行评估
if __name__ == "__main__":
    print("Starting evaluation...")
    results, field_stats, TP, FP, FN = evaluate_model(test_question_dataset, model, tokenizer)
    print_statistics(field_stats, results, TP, FP, FN)
    
    # 更新Excel文件
    excel_path = 'result.xlsx'
    update_excel_with_predictions(results, excel_path)







    
    
    
    
    
    







