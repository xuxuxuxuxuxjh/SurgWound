from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import base64
import io
from collections import defaultdict
import json
import re
from PIL import ImageDraw
import os

PROMPT = """
Detect all surgical wound in the image and return their locations in the form of coordinates. The format of output should be like {“bbox_2d”: [x1, y1, x2, y2]}.
"""

# 图片转base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def decode_base64_image(base64_string):
    """将base64编码的图像解码为PIL Image"""
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def Qwen_bbox(model, tokenizer, image_base64):
    # 解码图像
    image = decode_base64_image(image_base64)

    # 构建消息
    messages = [
        {
            "role": "system",
            "content": "You are a professional medical expert. Please detect all surgical wound in the image and return their locations in the form of coordinates."
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT}
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

    return response

def main():
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        'Qwen2.5-VL-7B-Instruct',
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoProcessor.from_pretrained(
        'Qwen/Qwen2.5-VL-7B-Instruct',
    )

    data_path = 'wound image path'
    save_path = 'cropped image path'
    os.makedirs(save_path, exist_ok=True)

    for file in sorted(os.listdir(data_path)):
        name = file.split('.')[0]
        if os.path.exists(f'{save_path}/{name}.jpg'):
            print(f'{name} already exists')
            continue
        print(name)
        # 读取jpg图片
        image_path = os.path.join(data_path, file)
        base64_image = image_to_base64(image_path)
        bbox_json_str = Qwen_bbox(model, tokenizer, base64_image)
        # 把string类型的json转换为dict
        clean_str = bbox_json_str.strip().replace("```json", "").replace("```", "").strip()
        try:
            bbox_json = json.loads(clean_str)
            image = decode_base64_image(base64_image)
            # 只保留crop区域
            for bbox in bbox_json:
                bbox_2d = bbox['bbox_2d']
                x1, y1, x2, y2 = bbox_2d
                print(x1, y1, x2, y2)
                cropped_image = image.crop((x1, y1, x2, y2))
                cropped_image.save(f'{save_path}/{name}.jpg')
                break

        except Exception as e:
            print(f'{name} clean_str: {clean_str}')
            print(f'{name} error: {e}')
            continue


if __name__ == "__main__":
    main()