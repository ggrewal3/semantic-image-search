"""Milestone 5: search a folder of images with a text query.

Usage:
    python src/search.py <image_directory> "<text_query>"

Example:
    python src/search.py images/ "a person working out"
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
MAX_IMAGES = 8
MAX_TOKENS = 77  # CLIP's text encoder cannot read sequences longer than this
TOP_K = 3  # how many results to show
EPSILON = 1e-8  # tiny number that protects against dividing by zero

if len(sys.argv) != 3:
    print('Usage: python src/search.py <image_directory> "<text_query>"')
    sys.exit(1)

image_dir = Path(sys.argv[1])
query = sys.argv[2]

if not image_dir.is_dir():
    print(f"Directory not found: {image_dir}")
    sys.exit(1)

if query.strip() == "":
    print("The search query is empty. Please provide some text.")
    sys.exit(1)

# --- Find and load the images (same approach as M2) ---

# Top level only, hidden files ignored, extensions matched case-insensitively.
image_paths = []
for path in image_dir.iterdir():
    if not path.is_file() or path.name.startswith("."):
        continue
    if path.suffix.lower() in SUPPORTED_EXTENSIONS:
        image_paths.append(path)

if len(image_paths) == 0:
    print(f"No .jpg, .jpeg or .png images found in: {image_dir}")
    sys.exit(1)

# Sort by filename so the order is the same on every run.
image_paths.sort(key=lambda path: path.name)

if len(image_paths) > MAX_IMAGES:
    print(
        f"Found {len(image_paths)} images. "
        f"Only the first {MAX_IMAGES} (sorted by filename) will be processed."
    )
    image_paths = image_paths[:MAX_IMAGES]

# filenames and images are parallel lists: filenames[i] is the name of
# images[i]. Later, row i of the embeddings and score i also belong to it.
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
    print("No valid images to search.")
    sys.exit(1)

# --- Load the model once ---

# The SAME model embeds the images and the text. That is what puts them in
# one shared space where they can be compared. PyTorch runs on the CPU by
# default, so no device setup is needed.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

# Evaluation mode: switches off training-only behavior such as dropout.
model.eval()

# --- Prepare the inputs ---

# Tokenize the query first and check its length, so a query that is too
# long is rejected before any model inference is run.
# truncation=False: never cut the user's text short without telling them.
text_inputs = processor(text=query, return_tensors="pt", truncation=False)

token_count = text_inputs["input_ids"].shape[1]
if token_count > MAX_TOKENS:
    print(
        f"Query is too long: {token_count} tokens, "
        f"but CLIP accepts at most {MAX_TOKENS} (including start and end tokens)."
    )
    print("Please use a shorter query.")
    sys.exit(1)

# All images in one batch: a tensor of shape (N, 3, 224, 224).
image_inputs = processor(images=images, return_tensors="pt")

# --- Generate the embeddings ---

# no_grad: we are only running the model, not training it, so PyTorch
# does not need to track gradients. This saves memory and time.
with torch.no_grad():
    image_outputs = model.get_image_features(
        pixel_values=image_inputs["pixel_values"],
    )
    text_outputs = model.get_text_features(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
    )

# The return type depends on the Transformers version. Older versions return
# the embedding tensor directly. Newer versions return an output object whose
# pooler_output field is the projected embedding.
if isinstance(image_outputs, torch.Tensor):
    image_embeddings = image_outputs
else:
    image_embeddings = image_outputs.pooler_output

if isinstance(text_outputs, torch.Tensor):
    text_features = text_outputs
else:
    text_features = text_outputs.pooler_output

# image_embeddings has shape (N, 512): one row per image.
# text_features has shape (1, 512). Take row 0 to get one (512,) vector.
text_embedding = text_features[0]

# --- Normalize every embedding to unit length ---

# Why: we want to compare the DIRECTION of the vectors, not their length.
# A vector divided by its own length has length 1 and the same direction.
# Once every vector has length 1, a plain dot product IS the cosine
# similarity, because the "divide by both norms" step from M4 has already
# been done.

# Length of each image embedding.
#   dim=1        -> add up across the 512 values of each row, so we get
#                   one length per image (not one total for the whole tensor).
#   keepdim=True -> keep the result as shape (N, 1) and not (N,), so each
#                   row of the (N, 512) tensor is divided by its own length.
# (torch.nn.functional.normalize(image_embeddings, dim=1) does all of this.)
image_norms = torch.sqrt(
    torch.sum(image_embeddings * image_embeddings, dim=1, keepdim=True)
)

# clamp(min=EPSILON) replaces any length smaller than EPSILON with EPSILON,
# so we can never divide by zero.
normalized_image_embeddings = image_embeddings / torch.clamp(image_norms, min=EPSILON)

# The text embedding is a single vector, so no dim or keepdim is needed.
text_norm = torch.sqrt(torch.sum(text_embedding * text_embedding))
normalized_text_embedding = text_embedding / torch.clamp(text_norm, min=EPSILON)

# --- Score every image against the query in one operation ---

# Matrix-vector multiplication: @ takes the dot product of EACH ROW of the
# (N, 512) matrix with the (512,) text vector. N rows give N dot products,
# so the result has shape (N,): one score per image, in the same order as
# filenames. Because every vector has length 1, each dot product is the
# cosine similarity between that image and the query.
scores = normalized_image_embeddings @ normalized_text_embedding

# --- Rank ---

# torch.sort returns two tensors: the scores from highest to lowest, and
# the original position (row number) of each one. The positions tell us
# which filename each sorted score belongs to.
sorted_scores, sorted_positions = torch.sort(scores, descending=True)

# Show the top 3, or fewer if there are fewer than 3 images.
result_count = min(TOP_K, len(filenames))

# --- Print the results ---

print()
print(f"Query: {query}")
print(f"Images processed: {len(filenames)}")
print()
print(f"Image embeddings:             {image_embeddings.shape}")
print(f"Normalized image embeddings:  {normalized_image_embeddings.shape}")
print(f"Text embedding:               {text_embedding.shape}")
print(f"Normalized text embedding:    {normalized_text_embedding.shape}")
print(f"Similarity scores:            {scores.shape}")
print()
print(f"{'Rank':<6}{'Image':<30}{'Similarity'}")
print("-" * 46)
for rank in range(result_count):
    # .item() turns a one-value tensor into a plain Python number.
    position = sorted_positions[rank].item()
    score = sorted_scores[rank].item()
    print(f"{rank + 1:<6}{filenames[position]:<30}{score:.4f}")

print()
# A score of 0.28 does NOT mean a 28% chance of a match. Scores are only
# meaningful when compared with each other.
print("Note: similarity scores are not probabilities.")
