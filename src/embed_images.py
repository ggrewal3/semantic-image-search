"""Milestone 2: turn UP TO 8 images into CLIP embeddings in one batch.

Usage:
    python src/embed_images.py images/
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
MAX_IMAGES = 8

if len(sys.argv) != 2:
    print("Usage: python src/embed_images.py images/")
    sys.exit(1)

image_dir = Path(sys.argv[1])
if not image_dir.is_dir():
    print(f"Directory not found: {image_dir}")
    sys.exit(1)

# Find the supported image files (top level only, hidden files ignored).
image_paths = []
unsupported_count = 0
for path in image_dir.iterdir():
    if not path.is_file() or path.name.startswith("."):
        continue
    if path.suffix.lower() in SUPPORTED_EXTENSIONS:
        image_paths.append(path)
    else:
        unsupported_count += 1

if unsupported_count > 0:
    print(f"Ignored {unsupported_count} file(s) with unsupported extensions.")

if len(image_paths) == 0:
    print(f"No .jpg, .jpeg or .png images found in: {image_dir}")
    sys.exit(1)

# Sort by filename so the order is the same on every run. The operating
# system does not promise any particular order when listing a folder.
image_paths.sort(key=lambda path: path.name)

if len(image_paths) > MAX_IMAGES:
    print(
        f"Found {len(image_paths)} images. "
        f"Only the first {MAX_IMAGES} (sorted by filename) will be processed."
    )
    image_paths = image_paths[:MAX_IMAGES]

# Load the images. filenames and images are parallel lists:
# filenames[i] is the name of images[i].
filenames = []
images = []
skipped_count = 0
for path in image_paths:
    try:
        # CLIP expects 3-channel RGB input.
        image = Image.open(path).convert("RGB")
    except Exception as error:
        print(f"Warning: skipping {path.name} (could not open: {error})")
        skipped_count += 1
        continue
    filenames.append(path.name)
    images.append(image)

print(f"Loaded {len(images)} image(s), skipped {skipped_count} due to loading errors.")

if len(images) == 0:
    print("No valid images to process.")
    sys.exit(1)

# Load the processor and model once, then reuse them for the whole batch.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

# Evaluation mode: switches off training-only behavior such as dropout.
model.eval()

# Batching: instead of running the model once per image, we stack all the
# images into one tensor and run the model a single time. The processor
# resizes every image to 224x224, so they can all be stacked together.
inputs = processor(images=images, return_tensors="pt")

# The input tensor has four dimensions: (N, 3, 224, 224)
#   N   = number of images in the batch
#   3   = color channels (red, green, blue)
#   224 = height in pixels
#   224 = width in pixels
print(f"Input tensor shape:  {inputs['pixel_values'].shape}")

# no_grad: we are only running the model, not training it, so PyTorch
# does not need to track gradients. This saves memory and time.
with torch.no_grad():
    outputs = model.get_image_features(pixel_values=inputs["pixel_values"])

# pooler_output has shape (N, 512): one row per image, and each row is that
# image's 512-number embedding. The rows are in the same order as the images
# we passed in, so row i belongs to filenames[i].
embeddings = outputs.pooler_output

for filename, embedding in zip(filenames, embeddings):
    print(f"{filename}: {embedding.shape}")

print(f"Output tensor shape: {embeddings.shape}")
