# Semantic Image Search

Search a folder of images with a plain-English description, using a pretrained CLIP model. No training, no captions and no filename matching: the query `"a person working out"` finds a photo of someone lifting weights because the text and the image land close together in CLIP's embedding space.

This is an educational Applied AI project. It was built in seven small milestones, and each milestone is a short standalone script that can be read from top to bottom. The code favours clarity over abstraction.

> **Status of the results in this README.** Every number below was measured by running the scripts in this repository on eight local photographs. Those photographs are **not** included in the repository, so the results cannot be reproduced exactly from a clone. See [Reproducing the evaluation](#reproducing-the-evaluation).

## Contents

1. [Project overview](#1-project-overview)
2. [Problem being solved](#2-problem-being-solved)
3. [Technical architecture](#3-technical-architecture)
4. [Technology stack](#4-technology-stack)
5. [Why CLIP](#5-why-clip)
6. [Installation](#6-installation)
7. [Repository structure](#7-repository-structure)
8. [Running each milestone](#8-running-each-milestone)
9. [Semantic search examples](#9-semantic-search-examples)
10. [Embeddings and cosine similarity](#10-embeddings-and-cosine-similarity)
11. [Evaluation methodology](#11-evaluation-methodology)
12. [Evaluation results](#12-evaluation-results)
13. [Performance benchmark](#13-performance-benchmark)
14. [Design decisions and tradeoffs](#14-design-decisions-and-tradeoffs)
15. [Known limitations](#15-known-limitations)
16. [Future improvements](#16-future-improvements)

## 1. Project overview

The project takes a folder of images and a text query, and returns the three images that best match the query. It does this in four steps:

1. Turn every image into a vector of 512 numbers (an *embedding*) with CLIP's vision encoder.
2. Turn the query into a vector of 512 numbers with CLIP's text encoder.
3. Measure how closely each image vector points in the same direction as the query vector (*cosine similarity*).
4. Sort the images by that score and show the top three.

Alongside the search script, the repository contains an evaluation script that measures retrieval quality on a labelled query set, and a benchmark that compares one-at-a-time inference with batched inference.

## 2. Problem being solved

Traditional image search depends on text attached to the image: filenames, tags or captions. A photo named `IMG_4032.png` is invisible to that kind of search, and a photo tagged "dog" will not be found by the query "a playful pet outdoors".

Semantic search removes the need for attached text. The model looks at the pixels and at the query, and compares their meaning directly. That makes untagged photo collections searchable, and lets a query use different words from any label a person would have chosen.

## 3. Technical architecture

```
  Images in a folder                          Text query
          |                                       |
          v                                       v
  Pillow: open, convert to RGB            CLIP tokenizer
          |                               (text -> token IDs)
          v                                       |
  CLIP image processor                            v
  (resize, crop to 224x224,               CLIP text encoder
   normalize pixel values)                + projection layer
          |                                       |
          v                                       v
  CLIP vision encoder                     Text embedding
  + projection layer                      shape (512,)
          |                                       |
          v                                       |
  Image embeddings                                |
  shape (N, 512)                                  |
          |                                       |
          +-------------------+-------------------+
                              |
                              v
               Normalize every vector to length 1
                              |
                              v
           Cosine similarity: one matrix-vector product
                   scores, shape (N,)
                              |
                              v
               Sort scores, highest first
                              |
                              v
                       Top 3 results
```

Both encoders belong to the same CLIP model, which was trained so that an image and a matching description produce vectors pointing in similar directions. That shared space is what makes the comparison meaningful.

## 4. Technology stack

| Component | Version | Role |
|---|---|---|
| Python | 3.13 | Language |
| PyTorch | 2.14.1 | Tensors and model inference |
| Hugging Face Transformers | 5.18.0 | CLIP model, tokenizer and image processor |
| Pillow | 12.3.0 | Opening image files |
| Model | `openai/clip-vit-base-patch32` | Pretrained vision-language model |

There are no other dependencies. Everything runs on the CPU.

## 5. Why CLIP

- **It embeds images and text into one space.** This is the property the whole project relies on. Most image models only embed images, and most text models only embed text.
- **It works without training.** CLIP was pretrained on a very large set of image-caption pairs, so it can match descriptions to photos it has never seen. Nothing in this project is trained or fine-tuned.
- **The base checkpoint is small enough to run locally.** `openai/clip-vit-base-patch32` is about 600 MB, needs no GPU, and embeds eight images in well under a second on a laptop (see the [benchmark](#13-performance-benchmark)).
- **It is well documented and widely used**, which matters for a project whose purpose is learning.

Larger CLIP checkpoints and newer vision-language models usually retrieve more accurately. They were not tested here.

## 6. Installation

These instructions are for macOS. Python 3.13 is assumed.

```bash
git clone https://github.com/ggrewal3/semantic-image-search.git
cd semantic-image-search

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Run `source .venv/bin/activate` again in each new terminal window. The first time any script runs, Transformers downloads the model (about 600 MB) into `~/.cache/huggingface`. Later runs use the cached copy.

Two messages printed on every run are harmless: a warning about unauthenticated requests to the Hugging Face Hub, and a "Loading weights" progress bar.

## 7. Repository structure

```
semantic-image-search/
├── README.md
├── requirements.txt
├── .gitignore
├── images/                    # put your own images here (not in the repository)
├── evaluation/
│   └── queries.json           # 20 labelled queries for the evaluation
├── docs/
│   └── experiment-report.md   # full write-up of the evaluation and benchmark
└── src/
    ├── embed_image.py         # M1: one image -> one embedding
    ├── embed_images.py        # M2: up to 8 images -> a batch of embeddings
    ├── embed_text.py          # M3: one text query -> one embedding
    ├── compare.py             # M4: cosine similarity of one image and one text
    ├── search.py              # M5: search a folder with a text query
    ├── evaluate.py            # M6: retrieval metrics on the labelled queries
    └── benchmark.py           # M7: individual vs batched inference timing
```

The `images/` folder is listed in `.gitignore`, so the repository contains the empty folder but no photographs.

## 8. Running each milestone

Every command is run from the project root with the virtual environment active. Each script needs at least one image in `images/`; the examples assume files such as `images/gym.png`.

| Milestone | What it shows | Command |
|---|---|---|
| M1 | Embed one image | `python src/embed_image.py images/gym.png` |
| M2 | Embed up to 8 images in one batch | `python src/embed_images.py images/` |
| M3 | Tokenize and embed one text query | `python src/embed_text.py "a person exercising in a gym"` |
| M4 | Cosine similarity of an image and a text | `python src/compare.py images/gym.png "a person exercising"` |
| M5 | Search a folder, show the top 3 | `python src/search.py images/ "a person working out"` |
| M6 | Evaluate retrieval on 20 labelled queries | `python src/evaluate.py images/ evaluation/queries.json` |
| M7 | Benchmark individual vs batched inference | `python src/benchmark.py images/` |

Notes:

- Text queries must be quoted so the shell passes them as one argument.
- Supported image formats are `.jpg`, `.jpeg` and `.png`. Only the top level of the folder is read.
- `embed_images.py` and `search.py` process at most the first 8 images, sorted by filename, and say so if there are more.
- CLIP reads at most 77 tokens of text. The scripts reject longer queries with an error; they never cut text short silently.
- M6 needs the eight specific filenames described under [Reproducing the evaluation](#reproducing-the-evaluation).

## 9. Semantic search examples

These are real outputs from `src/search.py` on the eight local test images (a beach, a car, a cat, a dog, a gym, a laptop, a mountain and a pizza). None of the queries uses a word from the matching filename.

```
$ python src/search.py images/ "a person working out"

Rank  Image                         Similarity
----------------------------------------------
1     gym.png                       0.2800
2     beach.png                     0.1977
3     cat.png                       0.1966
```

```
$ python src/search.py images/ "a place to go on holiday"

Rank  Image                         Similarity
----------------------------------------------
1     beach.png                     0.2627
2     mountain.png                  0.2304
3     car.png                       0.2074
```

```
$ python src/search.py images/ "a fast expensive vehicle"

Rank  Image                         Similarity
----------------------------------------------
1     car.png                       0.2576
2     dog.png                       0.1839
3     gym.png                       0.1832
```

```
$ python src/search.py images/ "something to eat"

Rank  Image                         Similarity
----------------------------------------------
1     pizza.png                     0.2257
2     cat.png                       0.2162
3     dog.png                       0.2105
```

The last example is worth a second look. The pizza is ranked first, but only 0.01 ahead of the cat. The ranking is right; the gap is small. Scores from this model sit in a narrow band, between 0.18 and 0.28 in these examples, so they are useful for ordering images against each other and not as absolute measures of a match.

The script also prints the tensor shapes at each stage; that part of the output is omitted above.

## 10. Embeddings and cosine similarity

**Embedding.** An embedding is a list of numbers that represents the meaning of an input. CLIP produces 512 numbers for an image and 512 for a piece of text. The individual numbers are not interpretable; what matters is how two embeddings relate to each other.

**Cosine similarity.** This measures whether two vectors point in the same direction, ignoring how long they are:

```
cosine_similarity(A, B) = dot(A, B) / (length(A) * length(B))
```

- `dot(A, B)` multiplies the two vectors position by position and adds up the results.
- `length(A)` is the square root of the sum of A's squared values.
- The result is between -1 and 1. Higher means more similar in direction.

**Normalization.** If every vector is first divided by its own length, each has length 1, and the formula reduces to a plain dot product. `search.py` does this, then scores all images in one operation:

```python
scores = normalized_image_embeddings @ normalized_text_embedding   # shape (N,)
```

`compare.py` computes the formula step by step and checks it against PyTorch's built-in `cosine_similarity`. For one test pair, the two agreed to within 3e-08.

**A similarity score is not a probability.** A score of 0.28 does not mean a 28% chance that the image matches. It only means something when compared with the scores of other images for the same query.

## 11. Evaluation methodology

`src/evaluate.py` runs 20 labelled queries against 8 images and measures how often the correct image is ranked highly.

**Dataset.** [evaluation/queries.json](evaluation/queries.json) holds 20 queries. Each has a hand-assigned list of relevant image filenames and a difficulty label:

| Difficulty | Queries | Style | Example |
|---|---|---|---|
| Easy | 8 | Names the subject directly | "a delicious pizza" |
| Medium | 7 | Uses a synonym or a broader category | "a domestic feline" |
| Hard | 5 | Abstract or indirect | "a portable device for programming" |

Eighteen queries have one relevant image. Two hard queries have two ("a peaceful place in nature" and "an animal commonly kept as a pet"). The labels were checked against the actual image contents before the evaluation was run, and were not changed afterwards.

**Metrics.** For each query, all 8 images are ranked by cosine similarity, then:

| Metric | Definition for one query | Reported as |
|---|---|---|
| Hit Rate@1 | 1 if the top-ranked image is relevant, otherwise 0 | Average over queries |
| Recall@3 | Relevant images in the top 3, divided by the number of relevant images | Average over queries |
| MRR (mean reciprocal rank) | 1 divided by the rank of the first relevant image | Average over queries |
| Top-1 margin | Highest score minus second-highest score | Average over queries |

The top-1 margin is a diagnostic for how clearly the winner stood out. It is not a measure of model confidence.

**Verification.** The metric functions operate on plain lists of filenames and are independent of the model. They were checked against hand-calculated rankings before being trusted. Details are in the [experiment report](docs/experiment-report.md).

### Reproducing the evaluation

The evaluation images are **not in this repository**. They are photographs collected locally, and they have not been published because their redistribution rights have not been confirmed. The results in the next section therefore cannot be independently reproduced from a clone.

To run the evaluator yourself, supply eight images of your own with exactly these filenames in `images/`:

| Filename | Subject of the original photograph |
|---|---|
| `beach.png` | Aerial view of a sandy beach and turquoise sea |
| `car.png` | A silver sports car parked on wet pavement at sunset |
| `cat.png` | An orange kitten sitting against a white background |
| `dog.png` | A golden retriever puppy lying on grass |
| `gym.png` | A man lifting dumbbells in a gym |
| `laptop.png` | An open laptop on a desk |
| `mountain.png` | A grassy mountain ridge at sunrise above clouds |
| `pizza.png` | A hand lifting a slice of pizza |

Then run:

```bash
python src/evaluate.py images/ evaluation/queries.json
```

The script validates the query file and stops with a clear message if an expected image is missing or unreadable.

**Your numbers will differ.** An embedding depends on the actual pixels, so different photographs of the same subjects give different similarity scores and margins, and possibly different rankings. Two of the queries describe details of the original photos ("a golden retriever puppy resting outdoors", "a sports car parked on pavement") and may fit other photos less well.

## 12. Evaluation results

Measured on 8 October 2026 with 8 images and 20 queries.

| Group | Queries | Hit Rate@1 | Recall@3 | MRR | Average top-1 margin |
|---|---|---|---|---|---|
| Overall | 20 | 1.0000 | 1.0000 | 1.0000 | 0.0538 |
| Easy | 8 | 1.0000 | 1.0000 | 1.0000 | 0.0829 |
| Medium | 7 | 1.0000 | 1.0000 | 1.0000 | 0.0420 |
| Hard | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0240 |

For all 20 queries, a relevant image was ranked first. There were no failure cases.

**How to read this.** These scores describe this one small test and should not be taken as the model's general retrieval accuracy:

- The dataset has only 8 images, assembled locally, each showing a clearly different subject. With one obvious answer per query and seven unrelated alternatives, the task is easy.
- The metrics have reached their maximum, so they cannot distinguish easy queries from hard ones, or this model from a better or worse one.
- This is not an industry benchmark and is not comparable to published retrieval results.

The average margin is the only number that moves. It falls from 0.0829 on easy queries to 0.0240 on hard ones, but part of that fall is expected for a reason unrelated to difficulty: two of the five hard queries have two correct images, and for both of them the top two results were the two correct images, with gaps of 0.0068 and 0.0013.

Per-query results and the supporting checks are in the [experiment report](docs/experiment-report.md).

## 13. Performance benchmark

`src/benchmark.py` embeds the same 8 images in two ways and times each:

- **Individual:** one processor call and one model call per image.
- **Batch:** one processor call and one model call for all 8 images.

Both timed regions include image preprocessing and the model forward pass. Model loading and reading image files from disk are excluded from both. Each approach gets 2 untimed warm-up runs and 10 timed runs, and the order alternates between runs so neither approach always goes first. Before timing, the script checks that the two approaches produce the same embeddings and stops if they do not.

**Measured result**

| | Individual | Batch |
|---|---|---|
| Median time for 8 images | 250.94 ms | 86.99 ms |
| Mean time for 8 images | 252.35 ms | 92.67 ms |
| Fastest and slowest run | 240.93 to 268.35 ms | 82.59 to 134.26 ms |
| Throughput (from median) | 31.88 images/second | 91.97 images/second |

**Speedup: 2.88x**, defined as median individual time divided by median batch time.

**Correctness:** both approaches returned embeddings of shape `(8, 512)`, and the maximum absolute difference between them was 0 (the values were identical). Small non-zero differences from floating-point rounding would also have been acceptable; the check allows 1e-4.

**Measurement conditions**

| | |
|---|---|
| Date | 8 October 2026 |
| Machine | Apple M4 Pro, macOS 15.5 |
| Device | CPU, 10 threads (PyTorch default) |
| Python / PyTorch | 3.13.9 / 2.14.1 |
| Workload | 8 PNG images, 2 warm-up and 10 timed runs per approach |

The table reports the first complete run of the benchmark. Two further runs immediately afterwards gave speedups of 2.74x and 2.76x, with batch medians of 76.50 ms and 80.41 ms.

These figures apply to this machine and this eight-image workload. CPU timings vary with background activity, temperature and thread scheduling, and the result says nothing about larger batches, other hardware or GPUs.

## 14. Design decisions and tradeoffs

| Decision | Reason | Cost |
|---|---|---|
| One standalone script per milestone, with repeated code | Each file can be read on its own, in order | Shared steps such as image loading are duplicated across files |
| Cosine similarity written out by hand | Shows the arithmetic; verified against PyTorch's built-in | A few more lines than calling the library function |
| Embeddings kept in memory only | Keeps the focus on the model and the math | Every run recomputes all image embeddings |
| A limit of 8 images in the search script | Keeps batches small and output readable while learning | Not usable on a real photo library as written |
| CPU only | No device-handling code; runs on any laptop | Slower than a GPU for larger collections |
| Reject queries over 77 tokens | Silent truncation would change the query's meaning without warning | Long queries must be shortened by the user |
| Exact search with `torch.sort` | Simple and exact for a handful of images | Compares the query with every image, which does not scale to very large collections |
| Evaluation stops on any unreadable image | Skipping an image would silently change what the metrics mean | Less forgiving than the search script, which skips bad files |
| Pinned dependency versions | Same behaviour on every install | Versions must be updated by hand |

One version-specific detail: in Transformers 5.x, `get_image_features` and `get_text_features` return an output object, and the projected embedding is its `pooler_output` field. Older tutorials index the result directly, which on this version returns per-patch or per-token vectors of the wrong shape.

## 15. Known limitations

- **The evaluation is small and easy.** Eight distinct images and 20 queries cannot expose many retrieval failures. See [Evaluation results](#12-evaluation-results).
- **The results are not publicly reproducible yet**, because the test images are not in the repository.
- **Labels were assigned by one person** and some are judgement calls. For example, the laptop's screen shows a landscape wallpaper, which could arguably match the nature queries.
- **The query set is not a held-out benchmark.** It was written with these eight images in mind.
- **Similarity scores are not probabilities** and are not comparable across different queries.
- **English only**, with short descriptive queries. Other languages and long or compositional queries were not tested.
- **No persistence.** Image embeddings are recomputed on every run.
- **No scaling work.** The search script is capped at 8 images and uses exact search.
- **One model checkpoint.** No other models or CLIP sizes were compared.
- **Benchmark results are for one machine** and one eight-image workload.

## 16. Future improvements

- Build a harder evaluation set with images that compete with each other (several dogs, several beaches) and with redistributable licences, so results can be reproduced.
- Add negative queries that match no image, to study what scores look like when there is no right answer.
- Save image embeddings to disk so a folder is only embedded once.
- Remove the 8-image limit by processing a large folder in several batches.
- Benchmark a range of batch sizes, and compare CPU with Apple's GPU backend (MPS).
- Compare `clip-vit-base-patch32` with larger CLIP checkpoints on the same evaluation.
- Add approximate nearest-neighbour search for collections too large to compare exhaustively.
- Add automated tests for the metric functions to the repository.
