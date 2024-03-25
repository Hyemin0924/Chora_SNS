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
import torchvision.transforms.functional as F
from torchvision.io import read_video, read_image, ImageReadMode
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from torchvision.io import read_video, read_image, ImageReadMode
from torchvision.utils import flow_to_image
import cv2
from torchvision.io import write_jpeg
import pickle
import argparse
import random

# Fast Parameter Modification! -----------------------

prompt = ""

input_folder_name = "Dataset"
output_folder_name = "Output_test1"
ref_folder_name = "Ref"

init_file = "init.png"
mask_file = "mask.png"

seed = 0

server = "http://localhost:7860"
# ----------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt', dest='prompt', default=prompt)
    parser.add_argument('--negative-prompt', dest='negative_prompt', default="")
    
    parser.add_argument('--init-image', dest='init_image', default=f"./{init_file}")
    parser.add_argument('--input-dir', dest='input_dir', default=f"./{input_folder_name}")
    parser.add_argument('--output-dir', dest='output_dir', default=f"./{output_folder_name}")
    parser.add_argument('--ref-dir', dest='ref_dir', default=f"./{ref_folder_name}")    

    parser.add_argument('--width', default=1360, type=int)
    parser.add_argument('--height', default=720, type=int)

    return parser.parse_args()

args = get_args()


def get_image_paths(folder):
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in image_extensions:
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)

y_paths = get_image_paths(args.input_dir)
z_paths = get_image_paths(args.ref_dir)

with open(args.init_image, "rb") as initial:
    startframe = base64.b64encode(initial.read()).decode("utf-8")
    
with open(mask_file, "rb") as mvsk:
        mask_image = base64.b64encode(mvsk.read()).decode("utf-8")

def get_controlnet_models():
    url = f"{server}/controlnet/model_list"

    depth_model = None
    depth_re = re.compile("^control_.*depth.* \[.{8}\]")

    response = requests.get(url)
    if response.status_code == 200:
        models = json.loads(response.content)
    else:
        raise Exception("Unable to list models from the SD Web API! "
                        "Is it running and is the controlnet extension installed?")

    for model in models['model_list']:
        if depth_model is None and depth_re.match(model):
            depth_model = model

    assert depth_model is not None, "Unable to find the depth model!  Ensure it's copied into the stable-diffusion-webui/extensions/models directory!" directory!"

    return depth_model


DEPTH_MODEL = get_controlnet_models()

def send_request(last_image_path, optical_flow_path, current_image_path, ref_image_path):
    url = f"{server}/sdapi/v1/img2img"
    
    with open(last_image_path, "rb") as b:
       last_image_encoded = base64.b64encode(b.read()).decode("utf-8")
    
    # Load and process the last image
    last_image = cv2.imread(last_image_path)
    last_image = cv2.cvtColor(last_image, cv2.COLOR_BGR2RGB)
    
    # Ref
    with open(ref_image_path, "rb") as b:
       reference_image = base64.b64encode(b.read()).decode("utf-8")
    

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
        "steps": 20,
        "cfg_scale": 4,
        "width": args.width,
        "height": args.height,
        "restore_faces": False,
        "tiling": False,
        "denoising_strength": 1.0,
        "override_settings": {},
        "override_settings_restore_afterwards": True,

        "init_images": [current_image],
        "resize_mode": 0,

        "mask_blur": 0,
        "inpainting_fill": 0,
        "inpaint_full_res": False,
        "inpaint_full_res_padding": 0,
        "inpainting_mask_invert": 0,

        "mask": mask_image,

        "alwayson_scripts": {
            "ControlNet":{
                "args": [
                    {
                        "input_image": current_image,
                        "module": "depth_midas",
                        "model": DEPTH_MODEL,
                        "weight": 1.0,
                        "guidance_end": 1,
                        "control_mode": 2,
                        "resize_mode": 0,
                    },
                ]
            }
        },
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

result = args.init_image
output_image_path = os.path.join(args.output_dir, f"output_image_0.png")

#with open(output_image_path, "wb") as f:
    #f.write(result)
    
last_image_path = args.init_image
for i in range(1, len(y_paths)):
    # Use the last image path and optical flow map to generate the next input
    optical_flow = infer(y_paths[i - 1], y_paths[i])
    
    # Modify your send_request to use the last_image_path
    result = send_request(last_image_path, optical_flow, y_paths[i], z_paths[i])
    data = json.loads(result)


    for j, encoded_image in enumerate(data["images"]):
        if j == 0:
            output_image_path = os.path.join(args.output_dir, f"output_image_{i}.png")
            last_image_path = output_image_path
        else:
            output_image_path = os.path.join(args.output_dir, f"controlnet_image_{j}_{i}.png")

        with open(output_image_path, "wb") as f:
           f.write(base64.b64decode(encoded_image))

    print(f"Written data for frame {i}:")
