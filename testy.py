from diffusers import StableDiffusionPipeline
import torch

# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Generate an image from text
prompt = "A futuristic cityscape at sunset"
image = pipe(prompt).images[0]

# Save the image
image.save("generated_image.png")