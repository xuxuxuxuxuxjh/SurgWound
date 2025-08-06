import json
import os
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import numpy as np
from bert_score import score as bert_score

file = ''

# 读取file
df = pd.read_excel(file)

data_path = '/fs/ess/PCON0023/xjh/data/wound_7_8/test_report_processed.json'
data_list = []
with open(data_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        data_list.append(data)

# 创建ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# 存储所有metrics
bleu_1_scores = []
bleu_2_scores = []
bleu_3_scores = []
rouge_1_scores = []
rouge_2_scores = []
rouge_l_scores = []
bert_scores = []

print("开始计算评估指标...")
print(f"总共有 {len(data_list)} 个样本")

# 为了提高BERTScore计算效率，先收集所有数据对
all_references = []
all_candidates = []
valid_indices = []

print("预处理数据...")
for i, data in enumerate(data_list):
    image_name = data.get('image_name', '')
    image_path = os.path.join('', image_name)
    question = data.get('question', '')
    answer = data.get('answer', '')

    # 读取df中image_name对应的report
    matching_rows = df.loc[df['image_name'] == image_name, 'report']
    if len(matching_rows) == 0:
        print(f"警告: 找不到图片 {image_name} 对应的报告，跳过...")
        continue
    
    report = matching_rows.values[0]
    if report.find('</think>') != -1:
        report = report.split('</think>')[-1]
    if report[0] == '[':
        report = report[1:-1]
    # 首尾单引号
    if report[0] == '\'':
        report = report[1:-1]
    # 首尾双引号
    if report[0] == '\"':
        report = report[1:-1]

    if i == 0:
        print(report)
    
    # 确保report和answer都是字符串
    if pd.isna(report) or pd.isna(answer):
        print(f"警告: 图片 {image_name} 的报告或答案为空，跳过...")
        continue
    
    report_str = str(report)
    answer_str = str(answer)
    
    all_references.append(report_str)
    all_candidates.append(answer_str)
    valid_indices.append(i)

print(f"有效样本数: {len(valid_indices)}")

# 批量计算BERTScore以提高效率
print("计算BERTScore...")
if len(all_references) > 0:
    P, R, F1 = bert_score(all_candidates, all_references, lang="en", verbose=True)
    bert_f1_scores = F1.tolist()
else:
    bert_f1_scores = []

print("计算其他指标...")
for idx, i in enumerate(valid_indices):
    data = data_list[i]
    image_name = data.get('image_name', '')
    answer = data.get('answer', '')
    
    report_str = all_references[idx]
    answer_str = all_candidates[idx]
    
    # 计算BLEU-1,BLEU-2,BLEU-3,ROUGE-1,ROUGE-2,ROUGE-L
    try:
        # BLEU scores
        report_tokens = [report_str.split()]
        answer_tokens = answer_str.split()
        
        bleu_1 = sentence_bleu(report_tokens, answer_tokens, weights=(1, 0, 0, 0))
        bleu_2 = sentence_bleu(report_tokens, answer_tokens, weights=(0.5, 0.5, 0, 0))
        bleu_3 = sentence_bleu(report_tokens, answer_tokens, weights=(0.33, 0.33, 0.33, 0))
        
        # ROUGE scores
        rouge_scores = scorer.score(report_str, answer_str)
        rouge_1 = rouge_scores['rouge1'].fmeasure
        rouge_2 = rouge_scores['rouge2'].fmeasure
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        # BERTScore (已经计算好)
        bert_f1 = bert_f1_scores[idx]
        
        # 存储分数
        bleu_1_scores.append(bleu_1)
        bleu_2_scores.append(bleu_2)
        bleu_3_scores.append(bleu_3)
        rouge_1_scores.append(rouge_1)
        rouge_2_scores.append(rouge_2)
        rouge_l_scores.append(rouge_l)
        bert_scores.append(bert_f1)
        
        # 打印进度
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(valid_indices)} 个样本")
            
    except Exception as e:
        print(f"计算指标时出错 (图片: {image_name}): {e}")
        continue

# 计算平均分数
if len(bleu_1_scores) > 0:
    avg_bleu_1 = np.mean(bleu_1_scores)
    avg_bleu_2 = np.mean(bleu_2_scores)
    avg_bleu_3 = np.mean(bleu_3_scores)
    avg_rouge_1 = np.mean(rouge_1_scores)
    avg_rouge_2 = np.mean(rouge_2_scores)
    avg_rouge_l = np.mean(rouge_l_scores)
    avg_bert_score = np.mean(bert_scores)
    
    print("\n" + "="*50)
    print("评估结果总结:")
    print("="*50)
    print(f"成功处理的样本数: {len(bleu_1_scores)}")
    print(f"BLEU-1 平均分数: {avg_bleu_1:.4f}")
    print(f"BLEU-2 平均分数: {avg_bleu_2:.4f}")
    print(f"BLEU-3 平均分数: {avg_bleu_3:.4f}")
    print(f"ROUGE-1 平均分数: {avg_rouge_1:.4f}")
    print(f"ROUGE-2 平均分数: {avg_rouge_2:.4f}")
    print(f"ROUGE-L 平均分数: {avg_rouge_l:.4f}")
    print(f"BERTScore 平均分数: {avg_bert_score:.4f}")
    print("="*50)
    
    # 保存详细结果到文件
    results_df = pd.DataFrame({
        'BLEU-1': bleu_1_scores,
        'BLEU-2': bleu_2_scores,
        'BLEU-3': bleu_3_scores,
        'ROUGE-1': rouge_1_scores,
        'ROUGE-2': rouge_2_scores,
        'ROUGE-L': rouge_l_scores,
        'BERTScore': bert_scores
    })
    
    # 添加统计信息
    results_summary = pd.DataFrame({
        'Metric': ['BLEU-1', 'BLEU-2', 'BLEU-3', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore'],
        'Mean': [avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_rouge_1, avg_rouge_2, avg_rouge_l, avg_bert_score],
        'Std': [np.std(bleu_1_scores), np.std(bleu_2_scores), np.std(bleu_3_scores), 
                np.std(rouge_1_scores), np.std(rouge_2_scores), np.std(rouge_l_scores), np.std(bert_scores)],
        'Min': [np.min(bleu_1_scores), np.min(bleu_2_scores), np.min(bleu_3_scores),
                np.min(rouge_1_scores), np.min(rouge_2_scores), np.min(rouge_l_scores), np.min(bert_scores)],
        'Max': [np.max(bleu_1_scores), np.max(bleu_2_scores), np.max(bleu_3_scores),
                np.max(rouge_1_scores), np.max(rouge_2_scores), np.max(rouge_l_scores), np.max(bert_scores)]
    })
    
    # 保存结果
    output_file = file.replace('.xlsx', '_evaluation_results.xlsx')
    with pd.ExcelWriter(output_file) as writer:
        results_summary.to_excel(writer, sheet_name='Summary', index=False)
        results_df.to_excel(writer, sheet_name='Detailed_Scores', index=False)
    
    print(f"详细结果已保存到: {output_file}")
    
else:
    print("错误: 没有成功处理任何样本!")


