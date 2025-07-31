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

image_path = ''


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

report_processed_list = []

print(len(report_processed_list))

path = 'train_report.json'

df = pd.read_excel('records_processed.xlsx')

prompt='''
<image>\nBased on the image, generate a detailed medical report that includes the following aspects: wound location, wound status, closure method, exudate characteristics, presence of erythema, presence of edema, urgency level, infection risk assessment.
'''

with open(path, 'r') as f:
    for line in f:
        data = json.loads(line)
        image_name = data['image_name']
        answer = data['answer']
        question = prompt
        row = df[df['Image Name'] == image_name].iloc[0]
        wound_location = row['Wound Location']
        closure_method = row['Closure Method']
        wound_status = row['Wound Status']
        exudate_characteristics = row['Exudate Characteristics']
        erythema = row['Erythema']
        edema = row['Edema']
        urgency_level = row['Urgency Level']
        infection_risk_assessment = row['Infection Risk Assessment']

        question += f"\nThe wound location is {wound_location}."
        question += f"\nThe closure method is {closure_method}."
        question += f"\nThe wound status is {wound_status}."
        question += f"\nThe exudate characteristics is {exudate_characteristics}."
        question += f"\nThe erythema is {erythema}."
        question += f"\nThe edema is {edema}."
        question += f"\nThe urgency level is {urgency_level}."
        question += f"\nThe infection risk assessment is {infection_risk_assessment}."

        report_processed_list.append(
        {
            "messages": [
                {
                    "content": question,
                    "role": "user"
                },
                {
                    "content": answer,
                    "role": "assistant"
                }
            ],
            "images": [
                image_path + data["image_name"]
            ]
        }
    )

print(len(report_processed_list))

with open('report.json', 'w') as f:
    json.dump(report_processed_list, f, indent=4, ensure_ascii=False)