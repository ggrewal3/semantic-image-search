"""Milestone 3: turn ONE text query into ONE CLIP embedding.

Usage:
    python src/embed_text.py "your search query"
"""

import sys

import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
MAX_TOKENS = 77  # CLIP's text encoder cannot read sequences longer than this

if len(sys.argv) != 2:
    print('Usage: python src/embed_text.py "your search query"')
    sys.exit(1)

query = sys.argv[1]
if query.strip() == "":
    print("The search query is empty. Please provide some text.")
    sys.exit(1)

# The processor wraps CLIP's tokenizer (for text) and image preprocessing.
# The model holds the pretrained weights. This is the same model as M1 and M2.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

# Evaluation mode: switches off training-only behavior such as dropout,
# so the same text always gives the same embedding.
model.eval()

# Tokenization: the model cannot read characters, so the tokenizer splits
# the text into pieces (tokens) from a fixed vocabulary and replaces each
# piece with its ID number in that vocabulary. It also adds a special
# start token at the front and a special end token at the back.
# truncation=False: never cut the user's text short without telling them.
inputs = processor(text=query, return_tensors="pt", truncation=False)

# Token IDs are NOT embeddings. They are just integer lookup numbers, like
# page numbers in a dictionary: ID 5222 is not "more" than ID 320.
# The model turns them into meaningful vectors later, during inference.
input_ids = inputs["input_ids"]

# The attention mask tells the model which positions hold real tokens (1)
# and which hold padding (0). Padding is only added when several texts of
# different lengths are batched together. With one query, every value is 1.
attention_mask = inputs["attention_mask"]

# Shape is (1, sequence_length): one query, and its length in tokens
# including the start and end tokens.
token_count = input_ids.shape[1]
if token_count > MAX_TOKENS:
    print(
        f"Query is too long: {token_count} tokens, "
        f"but CLIP accepts at most {MAX_TOKENS} (including start and end tokens)."
    )
    print("Please use a shorter query.")
    sys.exit(1)

# Turn each ID back into its text piece so we can see what it stands for.
# "</w>" marks the end of a word.
tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])

print(f"Text:                  {query}")
print(f"Token IDs:             {input_ids}")
print(f"Tokens:                {tokens}")
print(f"Token IDs shape:       {input_ids.shape}")
print(f"Attention mask:        {attention_mask}")
print(f"Attention mask shape:  {attention_mask.shape}")

# no_grad: we are only running the model, not training it, so PyTorch
# does not need to track gradients. This saves memory and time.
with torch.no_grad():
    # How CLIP builds the text embedding:
    #   1. The text encoder (a transformer) produces one vector per token.
    #   2. It keeps the vector at the end token, which summarizes the text.
    #   3. A projection layer maps that vector into the 512-dimensional
    #      space CLIP shares between text and images.
    outputs = model.get_text_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

# The return type depends on the Transformers version. Older versions return
# the embedding tensor directly. Newer versions return an output object whose
# pooler_output field is the projected embedding.
if isinstance(outputs, torch.Tensor):
    text_features = outputs
else:
    text_features = outputs.pooler_output

# Shape is (1, 512): one row per query. Take row 0 to get the single embedding.
embedding = text_features[0]

print(f"Type:                  {type(embedding)}")
print(f"Shape:                 {embedding.shape}")
print(f"First 10:              {embedding[:10]}")
