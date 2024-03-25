import os
import glob
import requests
import json
import cv2
import numpy as np
import re
import sys
import torch
from PIL import Image
from pprint import pprint
import base64
from io import BytesIO

import cv2
from torchvision.io import write_jpeg
import pickle
import argparse
import random

# Fast Parameter Modification! -----------------------

prompt = ""

input_folder_name = "Dataset"
output_folder_name = "Output"
# ref_folder_name = "Ref"

init_file = "./Dataset/카리나.png"

seed = 0

server = "http://127.0.0.1:7860"


# ----------------------------------------------------

def get_image_size(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt', dest='prompt', default=prompt)
    parser.add_argument('--negative-prompt', dest='negative_prompt', default="")

    parser.add_argument('--init-image', dest='init_image', default=f"./{init_file}")
    parser.add_argument('--input-dir', dest='input_dir', default=f"./{input_folder_name}")
    parser.add_argument('--output-dir', dest='output_dir', default=f"./{output_folder_name}")
    # parser.add_argument('--ref-dir', dest='ref_dir', default=f"./{ref_folder_name}")

    args = parser.parse_args()  # args 변수를 정의합니다.

    init_width, init_height = get_image_size(args.init_image)
    parser.add_argument('--width', default=init_width, type=int)
    parser.add_argument('--height', default=init_height, type=int)

    return parser.parse_args()


# 확인용
args = get_args()
print(args)


def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


y_paths = get_image_paths(args.input_dir)

def send_request(current_image_path):
    url = f"{server}/sdapi/v1/img2img"

    # Cur
    with open(current_image_path, "rb") as b:
        current_image = base64.b64encode(b.read()).decode("utf-8")

    data = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": seed,
        "subseed": 0,
        "subseed_strength": 0,

        "sampler_index": "DPM++ 2M Karras",
        "include_init_images": True,

        "batch_size": 1,
        "n_iter": 1,
        "steps": 50,
        "cfg_scale": 6,
        "width": args.width,
        "height": args.height,
        "restore_faces": False,
        "tiling": False,
        "denoising_strength": 0.3,
        "override_settings": {},
        "override_settings_restore_afterwards": True,

        "init_images": [current_image],
        "resize_mode": 0,
    }

    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.content
    else:
        try:
            error_data = response.json()
            print("Error:")
            print(str(error_data))

        except json.JSONDecodeError:
            print(f"Error: Unable to parse JSON error data.")
        return None


result = send_request(init_file)
data = json.loads(result)
for j, encoded_image in enumerate(data["images"]):
    output_image_path = os.path.join(args.output_dir, f"output_image_{j}.png")
    
with open(output_image_path, "wb") as f:
    f.write(base64.b64decode(encoded_image))
