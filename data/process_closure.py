import json
import random
import pandas as pd

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


path = ''

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
    question = '<image>\nBased on the image, ' + question
    formatted_question = question + "\nOptions:\n"
    for i, option in enumerate(options):
        formatted_question += f"{chr(65+i)}. {option}\n"
    formatted_question += f"Please answer with the only one choice ({', '.join([chr(ord('A') + i) + '. ' + opt for i, opt in enumerate(options)])}):"
    return formatted_question

def format_question_with_options_for_eval(question, options):
    """格式化问题和选项用于评估"""
    formatted_question = question + "\n\n"
    for i, option in enumerate(options):
        formatted_question += f"{chr(65+i)}. {option}\n"
    formatted_question += "\nPlease answer with the only one letter (A, B, C, D, etc.):"
    return formatted_question

closure_processed_list = []
image_path = ''
for data in closure_list:
    image_name = data['image_name'].split('.')[0].split('_')[0] + '.jpg'
    print(image_name)
    df = pd.read_excel('records_processed.xlsx')
    row = df[df['Image Name'] == image_name].iloc[0]
    wound_location = row['Wound Location']

    question = data['question']
    question = f"what is the closure method of this surgical wound?"

    options = data['options']
    random.shuffle(options)
    answer = data['answer']

    question_text = format_question_with_options(question, options)

    # 找到答案对应的字母
    answer_index = data["options"].index(data["answer"])
    answer_letter = chr(ord('A') + answer_index)
    answer_text = f"{answer_letter}. {data['answer']}"

    closure_processed_list.append(
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

print(len(closure_processed_list))

with open('closure.json', 'w') as f:
    json.dump(closure_processed_list, f, indent=4, ensure_ascii=False)