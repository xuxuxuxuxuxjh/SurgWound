#!/usr/bin/env python3
import json
import random
import os
from PIL import Image, ImageEnhance
import numpy as np

def augment_image(image_path, output_path, augmentation_type="random"):
    """
    对图片进行数据增强
    """
    try:
        image = Image.open(image_path)
        
        # 随机选择增强方法
        augmentations = []
        
        # 随机旋转 (-30 到 30 度)
        if random.choice([True, False]):
            angle = random.uniform(-30, 30)
            image = image.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        
        # 随机缩放 (0.8 到 1.2 倍)
        if random.choice([True, False]):
            scale = random.uniform(0.8, 1.2)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # 随机水平翻转
        if random.choice([True, False]):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        # 随机垂直翻转
        if random.choice([True, False]):
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        
        # 随机亮度调整 (0.8 到 1.2)
        if random.choice([True, False]):
            enhancer = ImageEnhance.Brightness(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        # 随机对比度调整 (0.8 到 1.2)
        if random.choice([True, False]):
            enhancer = ImageEnhance.Contrast(image)
            factor = random.uniform(0.8, 1.2)
            image = enhancer.enhance(factor)
        
        # 随机裁剪 (保留原图的90%-100%)
        if random.choice([True, False]):
            crop_ratio = random.uniform(0.9, 1.0)
            new_width = int(image.width * crop_ratio)
            new_height = int(image.height * crop_ratio)
            left = random.randint(0, image.width - new_width)
            top = random.randint(0, image.height - new_height)
            image = image.crop((left, top, left + new_width, top + new_height))
        
        # 保存增强后的图片
        image.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error augmenting image {image_path}: {e}")
        return False

if __name__ == "__main__":
    input_file = "train_question.json"
    output_file = "train_question_enhancement.json"
    image_path = "wound image path"

    # 存储所有数据
    all_data = []
    infection_risk_dict = {}
    
    low_count = 0
    medium_count = 0
    high_count = 0
    
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data = json.loads(line.strip())
                all_data.append(data)
                
                if data['field'] == 'Infection Risk Assessment':
                    infection_risk_dict[data['image_name']] = data['answer']

                    # 统计Infection Risk Assessment为Low, Medium, High的数量
                    if data['answer'] == 'Low':
                        low_count += 1
                    elif data['answer'] == 'Medium':
                        medium_count += 1
                    elif data['answer'] == 'High':
                        high_count += 1

    print(f"Original counts - Low: {low_count}, Medium: {medium_count}, High: {high_count}")

    # 进行数据增强
    enhanced_data = []
    
    # 获取需要增强的图片列表
    low_images = [img for img, risk in infection_risk_dict.items() if risk == 'Low']
    medium_images = [img for img, risk in infection_risk_dict.items() if risk == 'Medium']
    high_images = [img for img, risk in infection_risk_dict.items() if risk == 'High']
    
    print(f"Images to enhance - Low: {len(low_images)}, Medium: {len(medium_images)}, High: {len(high_images)}")
    
    # 确保输出目录存在
    os.makedirs(image_path, exist_ok=True)
    
    def create_enhanced_data_entries(original_image_name, enhanced_image_name, original_data_list):
        """为增强图片创建对应的数据条目"""
        enhanced_entries = []
        for data in original_data_list:
            if data['image_name'] == original_image_name:
                enhanced_data_entry = data.copy()
                enhanced_data_entry['image_name'] = enhanced_image_name
                enhanced_entries.append(enhanced_data_entry)
        return enhanced_entries

    # Low类图片扩充1倍（每个图片生成1个增强版本）
    print("Enhancing Low risk images...")
    for img_name in low_images:
        original_img_path = os.path.join(image_path, img_name)
        if os.path.exists(original_img_path):
            # 生成 xxx_2.jpg
            base_name = os.path.splitext(img_name)[0]
            ext = os.path.splitext(img_name)[1]
            enhanced_name = f"{base_name}_2{ext}"
            enhanced_path = os.path.join(image_path, enhanced_name)
            
            if augment_image(original_img_path, enhanced_path):
                # 创建对应的数据条目
                enhanced_entries = create_enhanced_data_entries(img_name, enhanced_name, all_data)
                enhanced_data.extend(enhanced_entries)
                print(f"Enhanced: {img_name} -> {enhanced_name}")
            else:
                print(f"Failed to enhance: {img_name}")
        else:
            print(f"Original image not found: {original_img_path}")
    
    # Medium类图片扩充2倍（每个图片生成2个增强版本）
    print("Enhancing Medium risk images...")
    for img_name in medium_images:
        original_img_path = os.path.join(image_path, img_name)
        if os.path.exists(original_img_path):
            base_name = os.path.splitext(img_name)[0]
            ext = os.path.splitext(img_name)[1]
            
            for i in [2, 3]:  # 生成 _2 和 _3 版本
                enhanced_name = f"{base_name}_{i}{ext}"
                enhanced_path = os.path.join(image_path, enhanced_name)
                
                if augment_image(original_img_path, enhanced_path):
                    # 创建对应的数据条目
                    enhanced_entries = create_enhanced_data_entries(img_name, enhanced_name, all_data)
                    enhanced_data.extend(enhanced_entries)
                    print(f"Enhanced: {img_name} -> {enhanced_name}")
                else:
                    print(f"Failed to enhance: {img_name}")
        else:
            print(f"Original image not found: {original_img_path}")
    
    # High类图片扩充4倍（每个图片生成4个增强版本）
    print("Enhancing High risk images...")
    for img_name in high_images:
        original_img_path = os.path.join(image_path, img_name)
        if os.path.exists(original_img_path):
            base_name = os.path.splitext(img_name)[0]
            ext = os.path.splitext(img_name)[1]
            
            for i in [2, 3, 4, 5]:  # 生成 _2, _3, _4, _5 版本
                enhanced_name = f"{base_name}_{i}{ext}"
                enhanced_path = os.path.join(image_path, enhanced_name)
                
                if augment_image(original_img_path, enhanced_path):
                    # 创建对应的数据条目
                    enhanced_entries = create_enhanced_data_entries(img_name, enhanced_name, all_data)
                    enhanced_data.extend(enhanced_entries)
                    print(f"Enhanced: {img_name} -> {enhanced_name}")
                else:
                    print(f"Failed to enhance: {img_name}")
        else:
            print(f"Original image not found: {original_img_path}")
    
    # 合并原始数据和增强数据
    final_data = all_data + enhanced_data
    
    # 统计最终的数量
    final_infection_counts = {'Low': 0, 'Medium': 0, 'High': 0}
    for data in final_data:
        if data['field'] == 'Infection Risk Assessment':
            if data['answer'] in final_infection_counts:
                final_infection_counts[data['answer']] += 1
    
    print(f"Final counts - Low: {final_infection_counts['Low']}, Medium: {final_infection_counts['Medium']}, High: {final_infection_counts['High']}")
    print(f"Total entries: {len(final_data)}")
    print(f"Enhanced entries: {len(enhanced_data)}")
    
    # 保存最终数据到输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in final_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Data enhancement completed. Output saved to: {output_file}")



