import streamlit as st
import torch
import torchvision
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Create a Streamlit sidebar
st.sidebar.title("Sentiment Classification")

# Create a text input field for the prompt
prompt = st.sidebar.text_input("Enter a sentence to classify its sentiment (type 'exit' to quit): ")

# If the user presses "Enter" or clicks outside of the text input field, classify the sentiment
if st.sidebar.button("Classify Sentiment"):
  if prompt == "exit":
    st.sidebar.warning("Exiting...")
    break

  # Generate an image of the classified sentiment
  image = pipe(prompt).images[0]

  # Save the image to disk
  image.save("output/" + prompt + ".png")

  # Show the image
  st.image("output/" + prompt + ".png")

# If the user presses "Exit", exit the app
if st.sidebar.button("Exit"):
  st.sidebar.warning("Exiting...")
  break
