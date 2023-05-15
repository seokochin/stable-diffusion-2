import gradio as gr
import torch
import torchvision
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

@gr.inputs(text="Enter a sentence to classify its sentiment (type 'exit' to quit)")
@gr.outputs(image="Output image")
def classify_sentiment(prompt):
  image = pipe(prompt).images[0]
  image.save("output/" + prompt + ".png")
  image.show()

gr.Interface(classify_sentiment).launch()
