# SurgWound-Bench: A Benchmark for Surgical Wound Diagnosis

The SurgWound dataset is publicly accessible at [huggingface](https://huggingface.co/datasets/xuxuxuxuxu/SurgWound).

This repository contains the official PyTorch implementation of the following paper:

> SurgWound-Bench: A Benchmark for Surgical Wound Diagnosis
>
> Jiahao Xu, Changchang Yin, Odysseas Chatzipanagiotou, Diamantis Tsilimigras, Kevin Clear, Bingsheng Yao, Dakuo Wang, Timothy Pawlik, Ping Zhang
>
> Abstract: Surgical site infection (SSI) is one of the most common and costly healthcare-associated infections and and surgical wound care remains a significant clinical challenge in preventing SSIs and improving patient outcomes. 
> While recent studies have explored the use of deep learning for preliminary surgical wound screening, progress has been hindered by concerns over data privacy and the high costs associated with expert annotation. Currently, no publicly available dataset or benchmark encompasses various types of surgical wounds, resulting in the absence of an open-source Surgical-Wound screening tool. To address this gap: (1) we present SurgWound, the first open-source dataset featuring a diverse array of surgical wound types. It contains 697 surgical wound images annotated by 3 professional surgeons with eight fine-grained clinical attributes. (2) Based on SurgWound, we introduce the first benchmark for surgical wound diagnosis, which includes visual question answering (VQA) and report generation tasks to comprehensively evaluate model performance. (3) Furthermore, we propose a three-stage learning framework, WoundQwen, for surgical wound diagnosis. In the first stage, we employ five independent MLLMs to accurately predict specific surgical wound characteristics. In the second stage, these predictions serve as additional knowledge inputs to two MLLMs responsible for diagnosing outcomes, which assess infection risk and guide subsequent interventions. In the third stage, we train a MLLM that integrates the diagnostic results from the previous two stages to produce a comprehensive report. This three-stage framework can analyze detailed surgical wound characteristics and provide subsequent instructions to patients based on surgical images, paving the way for personalized wound care, timely intervention, and improved patient outcomes.



## SurgWound Dataset and Benchmark

**SurgWound** is the first open-source dataset for surgical wound analysis across multiple procedure types.
SurgWound comprises 697 surgical wound images, each annotated by surgical experts at The Ohio State University Wexner Medical Center (OSWUMC).
Each image is accompanied by high-quality labels covering six surgical wound characteristic attributes and two diagnostic outcomes attributes.

**SurgWound-Bench** is the first multimodal benchmark for surgical wound analysis, which includes two tasks: SurgWound-VQA and SurgWound-Report

![](imgs\Architecture.png)



## WoundQwen: A Three-Stage Diagnostic Framework

 In the first stage, five of the six wound characteristics are predicted by specialized models: *Healing Status*, *Closure Method*, *Exudate Type*, *Erythema*, and *Edema*. *Location* is considered known clinical information and is not predicted. In the second stage, the wound image, together with the predicted characteristics and wound location, is input into two specialized models—WoundQwen\_risk for infection risk prediction and WoundQwen\_urgency for urgency level prediction. In the third stage, WoundQwen\_report utilizes the predictions from the first two stages along with the known location information to analyze images and generate a surgical wound report.

<img src="imgs\model.png" style="zoom:67%;" />



## Requirement

see `requirement.txt`

## Code Usage

We provide the relevant code for the WoundQwen three-stage diagnostic framework.

We adopt [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) as the training framework, employing supervised fine-tuning (SFT) with LoRA. The model architecture is based on [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), with [HuatuoGPT-Vision-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL) used as the base model.

**1.Preparation:**

Download the [SurgWound](https://huggingface.co/datasets/xuxuxuxuxu/SurgWound) dataset and related models: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and [HuatuoGPT-Vision-7B](https://huggingface.co/FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL).

**2.Data Augmentation:**

Apply random rotation, scaling, horizontal/vertical flipping, brightness and contrast adjustment, and cropping to augment surgical wound images.

Use Qwen2.5-VL-7B to generate cropped surgical wound images as additional augmentation data.

```
python ./data/enhancement.py
python ./data/Qwen_bbox.py
```

**3.Data Processing**

Convert VQA and report generation training data into the format required by LLaMA-Factory.

Adjust the proportion of each category to balance recognition performance between common and rare classes.

```
python ./data/process_closure.py
python ./data/process_status.py
python ./data/process_exudate.py
python ./data/process_erythema.py
python ./data/process_edema.py
python ./data/process_risk.py
python ./data/process_urgency.py
python ./data/process_report.py
```

**4.Training**

Add training data for `closure`, `status`, `exudate`, `erythema`, `edema`, `urgency`, and `report` to `/train/LLaMA-Factory/data/dataset_info.json`.

Modify `/train/LLaMA-Factory/examples/train_lora/qwen2_5vl_lora_sft.yaml` to set the correct model and training dataset configurations.

**5.Evaluation**

+ Surgical Wound Characteristics Evaluation

  ```
  python ./eval/eval_closure.py
  python ./eval/eval_status.py
  python ./eval/eval_exudate.py
  python ./eval/eval_erythema.py
  python ./eval/eval_edema.py
  ```

+ Surgical Wound Diagnostic Outcomes Evaluation

  Use the Stage 1 models (`WoundQwen_closure`, `WoundQwen_status`, `WoundQwen_exudate`, `WoundQwen_erythema`, `WoundQwen_edema`) to generate predictions. Save the prediction results to an Excel file for further processing.

  ```
  python ./eval/generation.py
  ```

  Based on Stage 1 predictions of surgical wound characteristics and known location information, generate and evaluate predictions for infection risk and urgency level.

  ```
  python ./eval/eval_risk.py
  python ./eval/eval_urgency.py
  python ./eval/risk_generation.py
  python ./eval/urgency_generation.py
  ```

+ Surgical Wound Report Generation Evaluation

  Run the following command to generate surgical wound reports:

  ```
  python ./eval/report_generation.py
  ```

  Use the following command to evaluate the generated reports：

  ```
  python ./eval/eval_report.py
  ```

  