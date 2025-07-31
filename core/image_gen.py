import os, torch
from diffusers import StableDiffusionPipeline

def generate_image(prompt, filename):
  out_dir = "assets/images"
  os.makedirs(out_dir, exist_ok=True)
  model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
  model = model.to("cuda" if torch.cuda.is_available() else "cpu")
  image = model(prompt).images[0]
  out_path = os.path.join(out_dir, filename)
  image.save(out_path)
  return out_path
