'''
export FLASK_APP=app
export FLASK_DEBUG=true
flask run

curl -X GET http://35.229.234.170:5000/ping 

curl -X GET http://35.229.234.170:5000/ping \
   -H 'Content-Type: application/json' \
   -d '{"expName":"drake-20","trainsetDir":"drake"}'

http://35.229.234.170:5000?prompt=A%20majestic%20lion%20jumping%20from%20a%20big%20stone%20at%20night
'''
from flask import Flask, request, send_file
import io
import torch
from torch import autocast
from diffusers import DiffusionPipeline
import json

app = Flask(__name__)
assert torch.cuda.is_available()

# TODO: add Optimum support
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.enable_model_cpu_offload()
#pipe.to("cuda")

refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

# Define how many steps and what % of steps to be run on each experts (80/20) here
n_steps = 40
high_noise_frac = 0.8


def run_inference(prompt):
  with autocast("cuda"):
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]  
  img_data = io.BytesIO()
  image.save(img_data, "PNG")
  img_data.seek(0)
  return img_data

@app.route('/')
def myapp():
    if "prompt" not in request.args:
        return "Please specify a prompt parameter", 400
    prompt = request.args["prompt"]
    img_data = run_inference(prompt)
    return send_file(img_data, mimetype='image/png')

@app.route("/ping")
def healthcheck():
    return json.dumps({"code": 200, "message": "responding"}).encode('utf-8')