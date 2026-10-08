"""Milestone 4: cosine similarity between ONE image and ONE text query.

Usage:
    python src/compare.py images/your_image.jpg "your text query"
"""

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_TOKENS = 77  # CLIP's text encoder cannot read sequences longer than this

if len(sys.argv) != 3:
    print('Usage: python src/compare.py images/your_image.jpg "your text query"')
    sys.exit(1)

image_path = Path(sys.argv[1])
query = sys.argv[2]

if not image_path.is_file():
    print(f"Image not found: {image_path}")
    sys.exit(1)

if query.strip() == "":
    print("The text query is empty. Please provide some text.")
    sys.exit(1)

try:
    # CLIP expects 3-channel RGB input.
    image = Image.open(image_path).convert("RGB")
except Exception as error:
    print(f"Could not open image {image_path.name}: {error}")
    sys.exit(1)

# Load the processor and model once. The SAME model embeds both the image
# and the text. That is what makes the two embeddings comparable.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

# Evaluation mode: switches off training-only behavior such as dropout.
model.eval()

# Image -> pixel tensor of shape (1, 3, 224, 224), as in M1.
image_inputs = processor(images=image, return_tensors="pt")

# Text -> token IDs and attention mask, as in M3.
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
    image_features = image_outputs
else:
    image_features = image_outputs.pooler_output

if isinstance(text_outputs, torch.Tensor):
    text_features = text_outputs
else:
    text_features = text_outputs.pooler_output

# Both have shape (1, 512). Take row 0 to get one 512-number vector each.
image_embedding = image_features[0]
text_embedding = text_features[0]

# --- Cosine similarity, step by step ---
#
#   cosine_similarity(A, B) = dot(A, B) / (norm(A) * norm(B))

# Dot product: multiply the two vectors position by position, then add up
# the 512 results. It is large when both vectors have big values in the
# same positions with the same sign.
# (torch.dot(image_embedding, text_embedding) gives the same number.)
dot_product = torch.sum(image_embedding * text_embedding)

# Norm (magnitude): the length of a vector. Square every value, add them
# up, and take the square root. This is Pythagoras in 512 dimensions.
# (torch.linalg.norm(image_embedding) gives the same number.)
image_norm = torch.sqrt(torch.sum(image_embedding * image_embedding))
text_norm = torch.sqrt(torch.sum(text_embedding * text_embedding))

# Why normalization matters: the dot product alone grows when either vector
# is longer, even if the two are no more alike. Dividing by both lengths
# removes that effect, so only the direction of the vectors is compared.
# The result is the cosine of the angle between them:
#    1 = same direction, 0 = unrelated (perpendicular), -1 = opposite.
manual_similarity = dot_product / (image_norm * text_norm)

# The same calculation using PyTorch's built-in function.
# dim=0 because each embedding is a 1-D tensor of 512 values.
pytorch_similarity = torch.nn.functional.cosine_similarity(
    image_embedding, text_embedding, dim=0
)

difference = torch.abs(manual_similarity - pytorch_similarity)

# .item() turns a one-value tensor into a plain Python number for printing.
print(f"Image:                      {image_path.name}")
print(f"Text:                       {query}")
print(f"Image embedding shape:      {image_embedding.shape}")
print(f"Text embedding shape:       {text_embedding.shape}")
print(f"Image embedding norm:       {image_norm.item():.4f}")
print(f"Text embedding norm:        {text_norm.item():.4f}")
print(f"Manual cosine similarity:   {manual_similarity.item():.8f}")
print(f"PyTorch cosine similarity:  {pytorch_similarity.item():.8f}")
print(f"Absolute difference:        {difference.item():.2e}")

# A cosine similarity is not a probability: 0.75 would NOT mean a 75% chance
# that the image matches the text. With CLIP, even good matches often score
# only around 0.2 to 0.3. Scores are useful for comparing against each other.
print("Note: this is a similarity score, not a probability.")
