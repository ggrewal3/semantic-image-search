"""Milestone 6: measure how well CLIP retrieval works on a labeled query set.

Usage:
    python src/evaluate.py <image_directory> <queries_json>

Example:
    python src/evaluate.py images/ evaluation/queries.json
"""

import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
MAX_TOKENS = 77  # CLIP's text encoder cannot read sequences longer than this
EXPECTED_QUERY_COUNT = 20
DIFFICULTIES = ["easy", "medium", "hard"]
TOP_K = 3
EPSILON = 1e-8  # tiny number that protects against dividing by zero


# --- Metric functions ---
#
# These work on plain Python lists of filenames and know nothing about CLIP
# or tensors. That keeps the metric math separate from the model, so it can
# be checked by hand with made-up rankings.
#
#   ranked_filenames: every image filename, best match first
#   relevant_images:  the filenames a person labeled as correct for the query


def hit_at_1(ranked_filenames, relevant_images):
    """1.0 if the top-ranked image is a relevant one, otherwise 0.0."""
    if ranked_filenames[0] in relevant_images:
        return 1.0
    return 0.0


def recall_at_k(ranked_filenames, relevant_images, k):
    """Fraction of the relevant images that appear in the top k results."""
    found = 0
    for filename in ranked_filenames[:k]:
        if filename in relevant_images:
            found += 1
    return found / len(relevant_images)


def first_relevant_rank(ranked_filenames, relevant_images):
    """Rank (1 = best) of the first relevant image in the ranking."""
    for position, filename in enumerate(ranked_filenames):
        if filename in relevant_images:
            return position + 1
    # Cannot happen when every image is ranked and the labels were validated.
    raise ValueError("No relevant image appears in the ranking.")


def reciprocal_rank(ranked_filenames, relevant_images):
    """1 / rank of the first relevant image: 1.0 for rank 1, 0.5 for rank 2..."""
    return 1.0 / first_relevant_rank(ranked_filenames, relevant_images)


def average(values):
    return sum(values) / len(values)


# --- Dataset validation ---


def validate_records(records, available_filenames):
    """Check the evaluation data. Returns a list of problems (empty if valid)."""
    if not isinstance(records, list):
        return ["The JSON file must contain a list of query records."]

    problems = []
    if len(records) != EXPECTED_QUERY_COUNT:
        problems.append(
            f"Expected exactly {EXPECTED_QUERY_COUNT} queries, found {len(records)}."
        )

    seen_queries = []
    for number, record in enumerate(records, start=1):
        label = f"Record {number}"

        if not isinstance(record, dict):
            problems.append(f"{label}: must be an object with query, relevant_images and difficulty.")
            continue

        query = record.get("query")
        if not isinstance(query, str) or query.strip() == "":
            problems.append(f"{label}: query must be a nonempty string.")
        elif query in seen_queries:
            problems.append(f"{label}: duplicate query '{query}'.")
        else:
            seen_queries.append(query)

        relevant_images = record.get("relevant_images")
        if not isinstance(relevant_images, list) or len(relevant_images) == 0:
            problems.append(f"{label}: relevant_images must be a nonempty list.")
        else:
            if len(set(map(str, relevant_images))) != len(relevant_images):
                problems.append(f"{label}: relevant_images contains a duplicate filename.")
            for filename in relevant_images:
                if filename not in available_filenames:
                    problems.append(
                        f"{label}: relevant image '{filename}' was not found in the image directory."
                    )

        difficulty = record.get("difficulty")
        if difficulty not in DIFFICULTIES:
            problems.append(
                f"{label}: difficulty must be easy, medium or hard (got '{difficulty}')."
            )

    return problems


# --- Report helpers ---


def print_metrics(title, results):
    """Print the four averaged metrics for a group of per-query results."""
    print(f"{title} ({len(results)} queries)")
    if len(results) == 0:
        print("  (no queries in this group)")
        return
    print(f"  Hit Rate@1:           {average([r['hit_at_1'] for r in results]):.4f}")
    print(f"  Recall@{TOP_K}:             {average([r['recall_at_k'] for r in results]):.4f}")
    print(f"  MRR:                  {average([r['reciprocal_rank'] for r in results]):.4f}")
    print(f"  Average top-1 margin: {average([r['margin'] for r in results]):.4f}")


def format_top_results(result):
    """Turn the top results into text like 'gym.png (0.2800), car.png (0.1900)'."""
    parts = []
    for filename, score in zip(result["top_filenames"], result["top_scores"]):
        parts.append(f"{filename} ({score:.4f})")
    return ", ".join(parts)


# --- Main program ---
#
# Why a main() function? In M1-M5 the code sat at the top level of the file,
# so it ran as soon as Python read the file. Here we want other code to be
# able to "import evaluate" and use the metric functions above WITHOUT
# loading CLIP and running a whole evaluation. Putting the program inside
# main() means it only runs when main() is called (see the bottom of the file).


def main():
    if len(sys.argv) != 3:
        print("Usage: python src/evaluate.py <image_directory> <queries_json>")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    queries_path = Path(sys.argv[2])

    if not image_dir.is_dir():
        print(f"Image directory not found: {image_dir}")
        sys.exit(1)

    if not queries_path.is_file():
        print(f"Evaluation file not found: {queries_path}")
        sys.exit(1)

    try:
        with open(queries_path, encoding="utf-8") as file:
            records = json.load(file)
    except json.JSONDecodeError as error:
        print(f"Evaluation file is not valid JSON: {error}")
        sys.exit(1)

    # --- Find the images (same rules as M5, but with no 8-image limit:
    # an evaluation must not silently leave images out) ---

    image_paths = []
    for path in image_dir.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(path)

    if len(image_paths) < 2:
        print(f"Need at least 2 images to rank, found {len(image_paths)} in: {image_dir}")
        sys.exit(1)

    # Sort by filename so the order is the same on every run.
    image_paths.sort(key=lambda path: path.name)
    filenames = [path.name for path in image_paths]

    # --- Validate the evaluation data before doing any expensive work ---

    problems = validate_records(records, filenames)
    if len(problems) > 0:
        print("The evaluation data is invalid:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    queries = [record["query"] for record in records]

    # --- Load the images ---

    # Unlike M5, an unreadable image is an error, not a warning. Skipping an
    # image here would quietly change what the metrics mean.
    images = []
    for path in image_paths:
        try:
            # CLIP expects 3-channel RGB input.
            images.append(Image.open(path).convert("RGB"))
        except Exception as error:
            print(f"Could not open image {path.name}: {error}")
            print("Fix or remove this file, then run the evaluation again.")
            sys.exit(1)

    # --- Load the model once ---

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Evaluation mode: switches off training-only behavior such as dropout.
    model.eval()

    # --- Tokenize all queries in one batch ---

    # Padding: a batch must be a rectangle, but the queries have different
    # lengths. padding=True adds filler tokens to the shorter queries so that
    # every row is as long as the longest query.
    # truncation=False: never cut a query short without telling the user.
    text_inputs = processor(
        text=queries,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    # Attention mask: 1 where a position holds a real token, 0 where it holds
    # padding. It tells the model to ignore the filler. Adding up each row
    # gives that query's real length, including its start and end tokens.
    token_counts = text_inputs["attention_mask"].sum(dim=1)

    too_long = False
    for query, token_count in zip(queries, token_counts.tolist()):
        if token_count > MAX_TOKENS:
            print(
                f"Query is too long: {token_count} tokens, but CLIP accepts at most "
                f"{MAX_TOKENS} (including start and end tokens): '{query[:60]}...'"
            )
            too_long = True
    if too_long:
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

    # The return type depends on the Transformers version. Older versions
    # return the embedding tensor directly. Newer versions return an output
    # object whose pooler_output field is the projected embedding.
    if isinstance(image_outputs, torch.Tensor):
        image_embeddings = image_outputs
    else:
        image_embeddings = image_outputs.pooler_output

    if isinstance(text_outputs, torch.Tensor):
        text_embeddings = text_outputs
    else:
        text_embeddings = text_outputs.pooler_output

    # image_embeddings: (number of images, 512)
    # text_embeddings:  (number of queries, 512)

    # --- Normalize every embedding to unit length (same math as M5) ---

    # dim=1 gives one length per row; keepdim=True keeps shape (rows, 1) so
    # that each row is divided by its own length.
    image_norms = torch.sqrt(
        torch.sum(image_embeddings * image_embeddings, dim=1, keepdim=True)
    )
    normalized_image_embeddings = image_embeddings / torch.clamp(image_norms, min=EPSILON)

    text_norms = torch.sqrt(
        torch.sum(text_embeddings * text_embeddings, dim=1, keepdim=True)
    )
    normalized_text_embeddings = text_embeddings / torch.clamp(text_norms, min=EPSILON)

    # --- Similarity matrix: every query against every image ---

    # .T is the transpose: it swaps rows and columns, turning the image
    # embeddings from shape (8, 512) into (512, 8).
    #
    # Matrix multiplication: (20, 512) @ (512, 8) gives (20, 8). The inner
    # sizes (512 and 512) must match and disappear; the outer sizes remain.
    # Entry [i, j] is the dot product of query i with image j. Because every
    # vector has length 1, that dot product is their cosine similarity.
    #
    # Row i    = the scores of query i against all images.
    # Column j = the scores for the image filenames[j].
    similarity_matrix = normalized_text_embeddings @ normalized_image_embeddings.T

    # --- Rank the images for each query ---

    # dim=1 sorts within each row, so every query gets its own ranking.
    # sorted_positions holds column numbers, which map back to filenames.
    # stable=True: if two scores were exactly equal, filename order decides.
    sorted_scores, sorted_positions = torch.sort(
        similarity_matrix, dim=1, descending=True, stable=True
    )

    # --- Compute the metrics for each query ---

    results = []
    for query_index, record in enumerate(records):
        positions = sorted_positions[query_index].tolist()
        scores = sorted_scores[query_index].tolist()
        ranked_filenames = [filenames[position] for position in positions]
        relevant_images = record["relevant_images"]

        results.append({
            "query": record["query"],
            "difficulty": record["difficulty"],
            "relevant_images": relevant_images,
            "top_filenames": ranked_filenames[:TOP_K],
            "top_scores": scores[:TOP_K],
            "first_relevant_rank": first_relevant_rank(ranked_filenames, relevant_images),
            "hit_at_1": hit_at_1(ranked_filenames, relevant_images),
            "recall_at_k": recall_at_k(ranked_filenames, relevant_images, TOP_K),
            "reciprocal_rank": reciprocal_rank(ranked_filenames, relevant_images),
            # Top-1 margin: how far the best score is ahead of the runner-up.
            # A diagnostic only. It is not a measure of model confidence.
            "margin": scores[0] - scores[1],
        })

    # --- Report ---

    print()
    print("=" * 70)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 70)
    print(f"Model:   {MODEL_NAME}")
    print(f"Images:  {len(filenames)}")
    print(f"Queries: {len(queries)}")
    print()
    print(f"Image embeddings:   {image_embeddings.shape}")
    print(f"Text embeddings:    {text_embeddings.shape}")
    print(f"Similarity matrix:  {similarity_matrix.shape}")
    print()

    print_metrics("OVERALL", results)
    print()
    for difficulty in DIFFICULTIES:
        group = [result for result in results if result["difficulty"] == difficulty]
        print_metrics(difficulty.upper(), group)
        print()

    print("-" * 70)
    print("INDIVIDUAL QUERY RESULTS")
    print("-" * 70)
    for number, result in enumerate(results, start=1):
        top_1_correct = "yes" if result["hit_at_1"] == 1.0 else "NO"
        print(f"{number:>2}. [{result['difficulty']}] {result['query']}")
        print(f"    Relevant:            {', '.join(result['relevant_images'])}")
        print(f"    Top {TOP_K}:               {format_top_results(result)}")
        print(f"    First relevant rank: {result['first_relevant_rank']}")
        print(f"    Top-1 correct:       {top_1_correct}")
        print(f"    Recall@{TOP_K}:            {result['recall_at_k']:.2f}")
        print(f"    Top-1 margin:        {result['margin']:.4f}")
        print()

    print("-" * 70)
    print("FAILURE CASES (top-ranked image is not a relevant image)")
    print("-" * 70)
    failures = [result for result in results if result["hit_at_1"] == 0.0]
    if len(failures) == 0:
        print("None.")
        print()
    for result in failures:
        print(f"Query:          {result['query']}  [{result['difficulty']}]")
        print(f"  Expected:     {', '.join(result['relevant_images'])}")
        print(f"  Actual top-1: {result['top_filenames'][0]}")
        print(f"  Actual top-{TOP_K}: {format_top_results(result)}")
        print()

    print("-" * 70)
    print("LIMITATIONS")
    print("-" * 70)
    print(f"- Small dataset: {len(filenames)} images and {len(queries)} queries. One query")
    print("  changes an overall metric by several points, and a difficulty")
    print("  group's metric by much more.")
    print("- Not yet reproducible from the public repository: the images are a")
    print("  small local dataset that is not included in the repository.")
    print("- Provisional labels: relevance was labeled by hand by one person, and")
    print("  some cases are judgement calls.")
    print("- Limited query diversity: short English descriptions of single objects")
    print("  and scenes.")
    print("- Similarity scores are not probabilities, and the top-1 margin is a")
    print("  diagnostic, not model confidence.")
    print(f"- Results on {len(filenames)} images do not establish performance on large datasets.")
    print("- The query set is not a held-out generalization benchmark.")


# The name guard. Python sets __name__ to "__main__" only for the file you
# run directly (python src/evaluate.py ...). When another file imports this
# one, __name__ is "evaluate" instead, so main() does not run and only the
# functions above become available.
if __name__ == "__main__":
    main()
