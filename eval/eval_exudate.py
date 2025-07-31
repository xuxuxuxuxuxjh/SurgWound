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
from peft import PeftModel


TARGET_ATTRS = [
    'Exudate Characteristics',
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


path = '/fs/ess/PCON0023/xjh/data/wound_7_8/test_question_processed.json'

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

exudate_processed_list = []
image_path = ''
for data in exudate_list:
    question = data['question']
    options = data['options']
    random.shuffle(options)
    answer = data['answer']

    question_text = format_question_with_options(question, options)

    # 找到答案对应的字母
    answer_index = data["options"].index(data["answer"])
    answer_letter = chr(ord('A') + answer_index)
    answer_text = f"{answer_letter}. {data['answer']}"

    exudate_processed_list.append(
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

print(len(exudate_processed_list))


test_path = 'test_question.json'
test_question_dataset = []
with open(test_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        if data['field'] == 'Exudate Characteristics':
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
                options = sample['options']
                random.shuffle(options)
                correct_answer = sample['answer']
                image_name = sample['image_name']
                image_path = os.path.join('', image_name)
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


    # 保存详细结果到文件
    # with open('eval_exudate.json', 'w', encoding='utf-8') as f:
    #     json.dump({
    #         'overall_accuracy': overall_accuracy,
    #         'field_statistics': dict(field_stats),
    #         'detailed_results': results
    #     }, f, ensure_ascii=False, indent=2)
    
    # print(f"\nDetailed results saved to 'eval_exudate.json'")

# 运行评估
if __name__ == "__main__":
    print("Starting evaluation...")
    results, field_stats, TP, FP, FN = evaluate_model(test_question_dataset, model, tokenizer)
    print_statistics(field_stats, results, TP, FP, FN)







    
    
    
    
    
    







