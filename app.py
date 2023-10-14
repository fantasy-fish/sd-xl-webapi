'''
export FLASK_APP=app
export FLASK_DEBUG=true
flask run

gunicorn -b :5000 --timeout=600 app:app

curl -X GET http://35.189.161.111:5000/ping 
curl -X GET http://35.189.161.111:5000/idle

curl -X GET http://35.189.161.111:5000/inference \
   -H 'Content-Type: application/json' \
   -d '{"prompt":"a photo of an astronaut riding a horse on mars","img_url":"https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"}' \
   -o output.png
'''
from flask import Flask, request, send_file
import io
import torch
from torch import autocast
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import json
from PIL import Image
import numpy as np
import subprocess as sp
from logging import exception


app = Flask(__name__)
assert torch.cuda.is_available()


# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# pipe.enable_model_cpu_offload()
# pipe.to("cuda")

# load both base & refiner
# base = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# )
# base.to("cuda")
# refiner = DiffusionPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0",
#     text_encoder_2=base.text_encoder_2,
#     vae=base.vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# )
# refiner.to("cuda")


# SDXL Img2Img only supports refiner now
# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLImg2ImgPipeline.__call__.example
# TODO: add Optimum support
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 20
high_noise_frac = 0.8

@app.route("/ping")
def healthcheck():
    return json.dumps({"code": 200, "message": "responding"}).encode('utf-8')

@app.route("/idle")
def get_gpu_utilization():
    cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"
    utilization = sp.check_output(cmd, shell=True)
    utilization = utilization.decode("utf-8").strip().split("\n")
    utilization = [int(x.replace(" %", "")) for x in utilization]
    return json.dumps({"code": 200, "message": str(utilization[0] <= 10)}).encode('utf-8')

def run_inference(prompt, init_image):
    # with torch.inference_mode():
    #     image = base(
    #         prompt=prompt,
    #         num_inference_steps=n_steps,
    #         denoising_end=high_noise_frac,
    #         output_type="latent",
    #         image=init_image
    #     ).images[0]
    #     final_image = refiner(
    #         prompt=prompt,
    #         num_inference_steps=n_steps,
    #         denoising_start=high_noise_frac,
    #         image=image,
    #     ).images[0]
    final_image = pipe(
        prompt=prompt,
        image=init_image,
    ).images[0]
    img_data = io.BytesIO()
    final_image.save(img_data, "PNG")
    img_data.seek(0)
    return img_data

@app.route('/inference')
def inference():
    if request.headers['Content-Type'] != 'application/json':
        exception("Header error")
        return json.dumps({"message":"Header error"}), 500
    try:
        content = request.get_json()
        prompt = content["prompt"]
        img_url = content["img_url"]
        init_image = load_image(img_url).convert("RGB")
        W, H = init_image.size
        if H > W:
            W = int(W * 1024 / H)
            H = 1024
        else:
            H = int(H * 1024 / W)
            W = 1024
        init_image = init_image.resize((W, H), Image.BILINEAR)
        img_data = run_inference(prompt, init_image)
        return send_file(img_data, mimetype='image/png')
    except Exception as e:
        exception("Inference process failed")
        return json.dumps({"message":"Inference process failed due to {}".format(e)}), 500
