"""Milestone 7: compare one-at-a-time and batched CLIP image inference.

Usage:
    python src/benchmark.py <image_directory>

Example:
    python src/benchmark.py images/
"""

import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png"]
WARMUP_RUNS = 2
TIMED_RUNS = 10
# Tolerances for deciding that two embeddings are "the same".
ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-4


def get_embeddings(outputs):
    """Return the projected embeddings from a get_image_features result."""
    # Older Transformers versions return the tensor directly. Newer versions
    # return an output object whose pooler_output field holds the embeddings.
    if isinstance(outputs, torch.Tensor):
        return outputs
    return outputs.pooler_output


# --- The two approaches being compared ---
#
# Both functions take the same already-opened images and return a tensor of
# shape (number of images, 512). Both do the same kinds of work: processor
# preprocessing, the model forward pass, and reading the embeddings.


def embed_individually(images, processor, model):
    """Approach A: one processor call and one model call PER IMAGE."""
    embeddings = []
    # no_grad: we are only running the model, not training it.
    with torch.no_grad():
        for image in images:
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
            embeddings.append(get_embeddings(outputs))
    # Each item has shape (1, 512). Join them into one (N, 512) tensor.
    return torch.cat(embeddings, dim=0)


def embed_batch(images, processor, model):
    """Approach B: one processor call and one model call for ALL images."""
    with torch.no_grad():
        inputs = processor(images=images, return_tensors="pt")
        outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
    return get_embeddings(outputs)


def time_one_run(embed_function, images, processor, model):
    """Run one approach once and return how many seconds it took."""
    # perf_counter() reads a high-resolution clock meant for measuring short
    # durations. The number itself is meaningless; only the difference
    # between two readings matters. It measures real elapsed ("wall-clock")
    # time, which is what a user waiting for results would experience.
    start = time.perf_counter()
    embed_function(images, processor, model)
    end = time.perf_counter()
    return end - start


def print_run_times(title, times):
    print(f"{title}:")
    for run_number, seconds in enumerate(times, start=1):
        print(f"  Run {run_number:>2}: {seconds * 1000:8.2f} ms")
    print()


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/benchmark.py <image_directory>")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    if not image_dir.is_dir():
        print(f"Image directory not found: {image_dir}")
        sys.exit(1)

    # --- Find and load the images (same rules as M6) ---

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

    # Images are opened ONCE, here, before any timing. Reading files from
    # disk is the same job for both approaches, and leaving it out keeps
    # disk speed and file caching from affecting the comparison.
    images = []
    for path in image_paths:
        try:
            images.append(Image.open(path).convert("RGB"))
        except Exception as error:
            print(f"Could not open image {path.name}: {error}")
            print("Fix or remove this file, then run the benchmark again.")
            sys.exit(1)

    image_count = len(images)

    # --- Load the model once ---

    # Model loading is excluded from the benchmark on purpose. It happens
    # once per program (and the download once per computer), however many
    # images are processed, and it is identical for both approaches. Timing
    # it would hide the difference we actually want to see.
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Evaluation mode: switches off training-only behavior such as dropout.
    # The model stays on the CPU, which is where PyTorch puts it by default.
    model.eval()

    # --- Warm-up ---

    # The first few runs of a model are slower than later ones: memory is
    # being allocated, CPU caches are cold and worker threads are starting.
    # We run each approach a few times and throw those timings away, so the
    # timed runs measure steady performance.
    for _ in range(WARMUP_RUNS):
        individual_embeddings = embed_individually(images, processor, model)
        batch_embeddings = embed_batch(images, processor, model)

    # --- Correctness check, BEFORE timing ---

    # A speed comparison is only meaningful if both approaches produce the
    # same answer, so we check that first and stop if they do not.
    # Tiny differences are normal: floating-point numbers are rounded after
    # every operation, and a batch may do the same arithmetic in a slightly
    # different order than a single image does.
    shapes_match = individual_embeddings.shape == batch_embeddings.shape
    if shapes_match:
        max_difference = torch.max(
            torch.abs(individual_embeddings - batch_embeddings)
        ).item()
        values_close = torch.allclose(
            individual_embeddings,
            batch_embeddings,
            atol=ABSOLUTE_TOLERANCE,
            rtol=RELATIVE_TOLERANCE,
        )
    else:
        max_difference = None
        values_close = False

    print()
    print("=" * 60)
    print("CLIP Inference Benchmark")
    print("=" * 60)
    print(f"Images:          {image_count}")
    print("Device:          CPU")
    print(f"CPU threads:     {torch.get_num_threads()} (PyTorch default)")
    print(f"Model:           {MODEL_NAME}")
    print(f"Python:          {platform.python_version()}")
    print(f"PyTorch:         {torch.__version__}")
    print(f"Warm-up runs:    {WARMUP_RUNS} per approach (not timed)")
    print(f"Timed runs:      {TIMED_RUNS} per approach")
    print()
    print("Correctness check (before timing)")
    print(f"  Individual embeddings shape:  {individual_embeddings.shape}")
    print(f"  Batch embeddings shape:       {batch_embeddings.shape}")
    print(f"  Shapes match:                 {shapes_match}")
    if max_difference is not None:
        print(f"  Maximum absolute difference:  {max_difference:.3e}")
    print(
        f"  Close (atol={ABSOLUTE_TOLERANCE}, rtol={RELATIVE_TOLERANCE}):"
        f"  {values_close}"
    )
    print()

    if not shapes_match or not values_close:
        print("The two approaches do NOT produce matching embeddings.")
        print("Timing them would be misleading, so the benchmark stops here.")
        sys.exit(1)

    # --- Timed runs ---

    # Why repeat? A single measurement is noisy: other programs, the
    # operating system and CPU temperature all change how long a run takes.
    # Several runs show how much the time varies and give a typical value.
    #
    # The order alternates (A then B, B then A, ...) so that neither approach
    # always goes first. If the computer slows down or speeds up during the
    # benchmark, both approaches are affected equally.
    individual_times = []
    batch_times = []
    for run_index in range(TIMED_RUNS):
        if run_index % 2 == 0:
            individual_times.append(time_one_run(embed_individually, images, processor, model))
            batch_times.append(time_one_run(embed_batch, images, processor, model))
        else:
            batch_times.append(time_one_run(embed_batch, images, processor, model))
            individual_times.append(time_one_run(embed_individually, images, processor, model))

    # --- Statistics ---

    # The mean is the ordinary average. The median is the middle value when
    # the times are sorted. One unusually slow run pulls the mean up but
    # barely moves the median, so the median is used for the headline numbers.
    individual_mean = statistics.mean(individual_times)
    batch_mean = statistics.mean(batch_times)
    individual_median = statistics.median(individual_times)
    batch_median = statistics.median(batch_times)

    # Speedup = median individual time / median batch time.
    # Above 1 means batching was faster; below 1 means it was slower.
    speedup = individual_median / batch_median

    # Throughput: how much work gets done per unit of time. Here that is
    # images embedded per second.
    individual_throughput = image_count / individual_median
    batch_throughput = image_count / batch_median

    # --- Report ---

    print(f"Each time below is for embedding all {image_count} images once.")
    print("Odd-numbered runs measured individual first; even-numbered runs")
    print("measured batch first.")
    print()
    print_run_times("Individual inference (one image per model call)", individual_times)
    print_run_times("Batch inference (all images in one model call)", batch_times)

    print("Summary")
    print(f"  Mean individual time:     {individual_mean * 1000:8.2f} ms")
    print(f"  Mean batch time:          {batch_mean * 1000:8.2f} ms")
    print(f"  Median individual time:   {individual_median * 1000:8.2f} ms")
    print(f"  Median batch time:        {batch_median * 1000:8.2f} ms")
    print(f"  Individual throughput:    {individual_throughput:8.2f} images/second")
    print(f"  Batch throughput:         {batch_throughput:8.2f} images/second")
    print()
    print("  Speedup = median individual time / median batch time")
    print(f"  Speedup:                  {speedup:8.2f}x")
    if speedup > 1:
        print("  Batch inference was faster in this run.")
    elif speedup < 1:
        print("  Individual inference was faster in this run.")
    else:
        print("  Both approaches took the same time in this run.")
    print()

    # Why batching CAN be faster: every model call has fixed overhead
    # (Python function calls, setting up each layer, starting worker
    # threads). One call for 8 images pays that overhead once, not 8 times,
    # and gives the CPU larger blocks of numbers that it can process in
    # parallel. It is not guaranteed: a large batch also needs more memory,
    # and on a CPU the gain can be small. That is why we measure.
    print("Limitations")
    print("  - CPU timings are noisy: background programs, CPU temperature and")
    print("    how the operating system schedules threads all affect them.")
    print("  - Results apply to this computer, this thread count and this")
    print(f"    workload of {image_count} images. Other machines will differ.")
    print(f"  - {image_count} images is a small batch. This does not show how batching")
    print("    behaves with larger batches or on a GPU.")
    print("  - Opening image files and loading the model are not timed.")


# Only run the benchmark when this file is run directly, not when imported.
if __name__ == "__main__":
    main()
