"""Milestone 1: turn ONE image into ONE CLIP embedding.

Usage:
    python src/embed_image.py images/your_image.jpg
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

if len(sys.argv) != 2:
    print("Usage: python src/embed_image.py images/your_image.jpg")
    sys.exit(1)

image_path = Path(sys.argv[1])
if not image_path.exists():
    print(f"Image not found: {image_path}")
    sys.exit(1)

# CLIP expects 3-channel RGB input. convert("RGB") handles PNGs with
# transparency, grayscale images, etc.
image = Image.open(image_path).convert("RGB")

# The processor does the preprocessing the model was trained with
# (resize, crop to 224x224, scale and normalize pixel values).
# The model holds the pretrained weights. Both are downloaded on first run.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

# Evaluation mode: switches off training-only behavior such as dropout,
# so the same image always gives the same embedding.
model.eval()

# return_tensors="pt" gives PyTorch tensors. The result contains
# "pixel_values" with shape (1, 3, 224, 224): a batch of one image.
inputs = processor(images=image, return_tensors="pt")

# no_grad: we are only running the model, not training it, so PyTorch
# does not need to track gradients. This saves memory and time.
with torch.no_grad():
    # Runs the vision encoder, then projects its output into the
    # embedding space that CLIP shares between images and text.
    outputs = model.get_image_features(pixel_values=inputs["pixel_values"])

# pooler_output is the image embedding: one vector summarizing the whole
# image. (outputs also has last_hidden_state, which holds one vector per
# image patch. We do not need that for search.)
image_features = outputs.pooler_output

# Shape is (1, 512): one row per image. Take row 0 to get the single embedding.
embedding = image_features[0]

print(f"Image:     {image_path.name}")
print(f"Type:      {type(embedding)}")
print(f"Shape:     {embedding.shape}")
print(f"First 10:  {embedding[:10]}")
