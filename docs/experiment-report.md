# Experiment Report: CLIP Text-to-Image Retrieval on a Small Local Dataset

**Date of measurements:** 8 October 2026
**Model:** `openai/clip-vit-base-patch32`
**Scripts:** `src/evaluate.py`, `src/benchmark.py`

## Summary

- On 8 local images and 20 labelled queries, a relevant image was ranked first for every query: Hit Rate@1, Recall@3 and MRR were all 1.0000.
- This test is too small and too easy to expose many retrieval failures. The perfect scores describe this dataset only and do not establish general retrieval accuracy.
- The average gap between the best and second-best score narrowed from 0.0829 on easy queries to 0.0240 on hard ones. Part of that narrowing comes from two hard queries that have two correct images each.
- Embedding the 8 images in one batch was 2.88 times faster than embedding them one at a time on the test machine, and produced identical embeddings.
- The image dataset is not in the repository, so these results are not publicly reproducible yet.

## 1. Research objective

Two questions were asked:

1. Does a pretrained CLIP model, used with no training, rank the correct image first for natural-language queries of increasing indirectness?
2. How much faster is batched image inference than one-at-a-time inference on a CPU, and do the two give the same embeddings?

A secondary aim was to confirm that the evaluation code itself is correct, independently of the model's output.

## 2. System architecture

```
Images -> CLIP vision encoder -> image embeddings  (8, 512)  --+
                                                               |--> normalize --> similarity matrix (20, 8)
Queries -> CLIP text encoder -> text embeddings   (20, 512)  --+                         |
                                                                                         v
                                                                    sort each row, highest first
                                                                                         |
                                                                                         v
                                                                           metrics per query
```

Each embedding is the encoder's output after CLIP's projection layer, which maps images and text into one shared 512-dimensional space. Every embedding is divided by its own length, so the matrix product of text embeddings with the transposed image embeddings gives the cosine similarity of every query with every image. Row *i* of that matrix holds the scores for query *i*, and column *j* corresponds to the *j*-th image in filename order.

## 3. CLIP checkpoint

| Property | Value |
|---|---|
| Checkpoint | `openai/clip-vit-base-patch32` |
| Vision encoder | Vision Transformer, base size, 32-pixel patches, 224x224 input |
| Embedding size | 512 |
| Maximum text length | 77 tokens, including start and end tokens |
| Training in this project | None; the pretrained weights are used unchanged |
| Libraries | PyTorch 2.14.1, Transformers 5.18.0, Pillow 12.3.0, Python 3.13.9 |
| Device | CPU |

## 4. Dataset description

**Images.** Eight PNG photographs, collected locally, one per subject:

| Filename | Content |
|---|---|
| `beach.png` | Aerial view of a sandy beach and turquoise sea, with one small figure |
| `car.png` | A silver sports car parked on wet pavement under a sunset sky |
| `cat.png` | An orange kitten sitting against a white background |
| `dog.png` | A golden retriever puppy lying on grass |
| `gym.png` | A man lifting dumbbells in a gym |
| `laptop.png` | An open laptop on a desk; its screen shows a lakeshore-and-mountain wallpaper |
| `mountain.png` | A grassy mountain ridge at sunrise above a sea of clouds |
| `pizza.png` | A hand lifting a slice of pizza |

The subjects are deliberately distinct. No two images show the same kind of thing, apart from the cat and dog both being pets and the beach and mountain both being landscapes.

**Queries.** Twenty queries in `evaluation/queries.json`, each with a list of relevant filenames and a difficulty label: 8 easy (names the subject), 7 medium (synonym or broader category) and 5 hard (abstract or indirect). Eighteen queries have one relevant image and two have two.

**Labelling.** Relevance labels were first derived from the filenames, then checked by viewing each image before the evaluation was run. Two queries were reworded to describe what the photos actually show ("a golden retriever puppy resting outdoors", "a sports car parked on pavement"). No label was changed after seeing model output.

**Availability.** The images are excluded from the repository because their redistribution rights have not been confirmed. Only the query file is published.

## 5. Evaluation methodology

1. Validate the query file: exactly 20 records, non-empty unique queries, non-empty relevance lists without duplicates, every referenced file present, and a valid difficulty.
2. Load all 8 images. An unreadable image stops the run; nothing is skipped.
3. Embed the 8 images in one batch.
4. Tokenize the 20 queries in one batch with padding and without truncation. Reject any query over 77 tokens.
5. Embed the 20 queries in one batch.
6. Normalize all embeddings and compute the 20 x 8 similarity matrix.
7. Sort each row from highest to lowest score, using a stable sort so that an exact tie would fall back to filename order.
8. Compute the metrics for each query, then average overall and by difficulty.

The run is deterministic: the same images and queries give the same scores on the same machine.

## 6. Evaluation metrics

| Metric | Per-query definition | Range |
|---|---|---|
| Hit Rate@1 | 1 if the top-ranked image is relevant, otherwise 0 | 0 or 1 |
| Recall@3 | Relevant images in the top 3 divided by the total number of relevant images | 0 to 1 |
| Reciprocal rank | 1 divided by the rank of the first relevant image | 1/8 to 1 |
| Top-1 margin | Highest similarity minus second-highest similarity | 0 or more |

Each is averaged over queries; the averaged reciprocal rank is the MRR. For the 18 single-answer queries, Recall@3 can only be 0 or 1. The top-1 margin is computed for every query whether or not the top result is correct, and is treated as a diagnostic, not as model confidence.

## 7. Evaluation results

Tensor shapes were as expected: image embeddings `(8, 512)`, text embeddings `(20, 512)`, similarity matrix `(20, 8)`.

| Group | Queries | Hit Rate@1 | Recall@3 | MRR | Average top-1 margin |
|---|---|---|---|---|---|
| Overall | 20 | 1.0000 | 1.0000 | 1.0000 | 0.0538 |
| Easy | 8 | 1.0000 | 1.0000 | 1.0000 | 0.0829 |
| Medium | 7 | 1.0000 | 1.0000 | 1.0000 | 0.0420 |
| Hard | 5 | 1.0000 | 1.0000 | 1.0000 | 0.0240 |

All 20 queries retrieved a relevant image at rank 1.

**Per-query results**

| # | Difficulty | Query | Top 1 (score) | Runner-up (score) | Margin |
|---|---|---|---|---|---|
| 1 | easy | a person exercising in a gym | gym.png (0.2634) | dog.png (0.1882) | 0.0752 |
| 2 | medium | someone doing a workout | gym.png (0.2597) | dog.png (0.2085) | 0.0511 |
| 3 | hard | physical fitness and training | gym.png (0.2596) | dog.png (0.2143) | 0.0453 |
| 4 | easy | a golden retriever puppy resting outdoors | dog.png (0.2801) | cat.png (0.1650) | 0.1150 |
| 5 | medium | a domestic canine animal | dog.png (0.2742) | cat.png (0.2397) | 0.0345 |
| 6 | hard | a playful pet outdoors | dog.png (0.2487) | cat.png (0.2303) | 0.0184 |
| 7 | easy | a cat sitting | cat.png (0.2641) | dog.png (0.1852) | 0.0789 |
| 8 | medium | a domestic feline | cat.png (0.2581) | dog.png (0.1951) | 0.0630 |
| 9 | easy | a delicious pizza | pizza.png (0.2699) | cat.png (0.1797) | 0.0901 |
| 10 | medium | Italian food for dinner | pizza.png (0.2328) | car.png (0.2031) | 0.0297 |
| 11 | easy | a sports car parked on pavement | car.png (0.2311) | gym.png (0.1639) | 0.0672 |
| 12 | medium | a motor vehicle | car.png (0.2458) | cat.png (0.1986) | 0.0473 |
| 13 | easy | a sandy beach | beach.png (0.2870) | car.png (0.2071) | 0.0799 |
| 14 | medium | a relaxing coastal landscape | beach.png (0.2703) | mountain.png (0.2464) | 0.0239 |
| 15 | easy | a mountain landscape | mountain.png (0.2635) | beach.png (0.2140) | 0.0494 |
| 16 | medium | a beautiful mountainous region | mountain.png (0.2599) | beach.png (0.2154) | 0.0445 |
| 17 | easy | a laptop computer | laptop.png (0.2839) | cat.png (0.1768) | 0.1071 |
| 18 | hard | a portable device for programming | laptop.png (0.2305) | cat.png (0.1824) | 0.0481 |
| 19 | hard | a peaceful place in nature | beach.png (0.2505) | mountain.png (0.2437) | 0.0068 |
| 20 | hard | an animal commonly kept as a pet | dog.png (0.2677) | cat.png (0.2664) | 0.0013 |

**Interpretation.** With eight clearly different images, each query has one obvious answer and seven unrelated alternatives. The retrieval metrics have reached their maximum and can no longer separate easy from hard queries, or this model from another. The result shows that the pipeline works end to end on this data. It is not a general accuracy claim about the model and is not comparable to published retrieval benchmarks.

## 8. Analysis of score margins

Since the retrieval metrics are saturated, the margin is the only measurement that varies.

- **Winning scores sit in a narrow band.** The top score across the 20 queries ranged from 0.2305 to 0.2870. Raw CLIP similarities are small numbers, and a correct match does not approach 1.
- **Margins shrink as queries become less direct.** The average margin was 0.0829 for easy queries, 0.0420 for medium and 0.0240 for hard.
- **The hard-query average is pulled down by two multi-answer queries.** Queries 19 and 20 each have two relevant images, and in both cases the top two results were those two images, with margins of 0.0068 and 0.0013. A small gap between two correct answers is unsurprising and says nothing about difficulty. Averaging only the three single-answer hard queries gives 0.0373, close to the medium average.
- **The closest single-answer calls** were query 6 (dog ahead of cat by 0.0184) and query 14 (beach ahead of mountain by 0.0239). In both, the runner-up is the image most related to the correct one.
- **The largest margins** belonged to queries that describe the photo specifically: query 4 (0.1150) and query 17 (0.1071).

With 5 to 8 queries per difficulty group, these averages are indicative only. No statistical significance is claimed.

## 9. Individual versus batched inference benchmark

**Design.** `src/benchmark.py` embeds the same 8 images two ways:

- **Individual:** for each image, one processor call and one model call.
- **Batch:** one processor call and one model call for all 8 images.

Measures taken to keep the comparison fair:

- Both timed regions contain image preprocessing, the model forward pass and reading the embeddings.
- Images are opened and converted to RGB once, before any timing. Model loading is also outside the timed regions.
- Each approach runs twice untimed as a warm-up, then 10 times with `time.perf_counter()`.
- The order alternates: odd-numbered runs time individual first, even-numbered runs time batch first.
- Both use the same model, CPU, PyTorch's default thread count and `torch.no_grad()`.
- Correctness is checked before timing, and the benchmark stops if the embeddings disagree.

**Conditions.** Apple M4 Pro, macOS 15.5, CPU with 10 threads (PyTorch default), Python 3.13.9, PyTorch 2.14.1, measured on 8 October 2026.

**Times for embedding all 8 images once, in milliseconds**

| Run | Individual | Batch |
|---|---|---|
| 1 | 240.93 | 93.89 |
| 2 | 250.56 | 134.26 |
| 3 | 251.32 | 82.59 |
| 4 | 268.35 | 90.17 |
| 5 | 256.95 | 86.15 |
| 6 | 259.84 | 85.42 |
| 7 | 243.41 | 86.20 |
| 8 | 245.79 | 83.85 |
| 9 | 250.25 | 96.35 |
| 10 | 256.10 | 87.77 |

| Statistic | Individual | Batch |
|---|---|---|
| Mean | 252.35 ms | 92.67 ms |
| Median | 250.94 ms | 86.99 ms |
| Throughput (8 images / median) | 31.88 images/second | 91.97 images/second |

**Speedup = median individual time / median batch time = 2.88x.**

Batch run 2 (134.26 ms) is an outlier about 50% above the batch median. It raises the batch mean by several milliseconds but barely affects the median, which is why the headline figures use medians.

**Repeatability.** The figures above are from the first complete run of the benchmark. Two further runs immediately afterwards gave:

| Benchmark run | Median individual | Median batch | Speedup |
|---|---|---|---|
| First (reported above) | 250.94 ms | 86.99 ms | 2.88x |
| Second | 209.94 ms | 76.50 ms | 2.74x |
| Third | 222.00 ms | 80.41 ms | 2.76x |

Absolute times moved by up to about 16% between runs, while the speedup stayed between 2.74x and 2.88x.

**Scope.** These results apply to this machine, this thread count and an eight-image workload. They do not show how batching behaves at larger batch sizes, on other CPUs or on a GPU. The benchmark does not separate preprocessing time from model time, so it does not show which of the two accounts for the difference.

## 10. Correctness validation

Four checks were made that do not depend on the retrieval results looking plausible.

**Metric functions against hand-calculated examples.** The metric functions take plain lists of filenames, so they were tested with made-up rankings of eight items and no model. Ten ranking cases were checked, including single relevant items at ranks 1, 2, 3, 4 and 8, and pairs of relevant items at ranks (1, 2), (1, 5), (3, 4), (5, 6) and (2, 3). Averages over four queries at ranks 1, 2, 4 and 8 gave Hit Rate@1 0.25, Recall@3 0.5 and MRR 0.46875, matching the hand calculation. All checks passed.

**Failure reporting with deliberately wrong labels.** Because the real data produced no failures, a copy of the query file was made with four labels changed to wrong images. The evaluator listed exactly those four queries as failures and reported Hit Rate@1 0.8000, Recall@3 0.9250 and MRR 0.8875. These match the values worked out by hand: 16/20, 18.5/20 and 17.75/20.

**Batched versus individual text embeddings.** In the batch of 20 queries, 19 were padded up to the longest query's 9 tokens. Each query was also embedded on its own, with no padding, and the two sets were compared:

| Comparison | Result |
|---|---|
| Largest absolute difference in any embedding value | 3.8e-06 |
| Lowest cosine similarity between a query's two embeddings | 0.9999997 |
| Equal within a tolerance of 1e-5 | Yes |
| Largest difference in any similarity score | 2.1e-07 |
| Rankings for all 20 queries | Identical |

Batched text embeddings therefore agreed with individually generated ones within numerical tolerance. The differences are consistent with 32-bit floating-point rounding. This check was worth making because CLIP's padding token has the same ID as its end-of-text token, and the text embedding is taken from the end-of-text position.

**Batched versus individual image embeddings.** In the benchmark, both approaches returned shape `(8, 512)` and the maximum absolute difference was 0: the values were bit-for-bit identical. To confirm the check can fail, the batch was also run with the images in reversed order, which gave a maximum difference of 3.75 and failed the tolerance test, as it should.

In addition, four of the evaluation queries were run through `src/search.py`, which embeds one query at a time through separate code. Its top-3 results and scores matched the evaluator's to four decimal places.

## 11. Failure analysis

There were no failure cases: no query had a non-relevant image ranked first, and no relevant image fell outside the top 3.

Two things were watched for and did not occur:

- **The laptop as a distractor.** The laptop's screen shows a landscape, which could plausibly attract the nature queries (13, 14, 15, 16 and 19). `laptop.png` did not appear in the top 3 for any of them.
- **Confusion between related images.** Dog and cat, and beach and mountain, were each other's runner-up for most related queries, but the labelled image was always ahead.

The absence of failures is a property of this dataset more than evidence about the model. No conclusions about typical CLIP failure modes can be drawn from it.

## 12. Limitations

- **Small dataset.** Eight images and 20 queries. One query is worth 5 points of an overall metric, and 12.5 to 20 points of a difficulty group's metric.
- **Distinct subjects.** The images do not compete with each other, so the task does not test fine-grained discrimination.
- **Not publicly reproducible.** The images are not in the repository. Anyone repeating the evaluation with their own photographs will get different scores.
- **Provisional labels.** Relevance was judged by one person. Some cases, such as whether the laptop's wallpaper makes it relevant to a nature query, are judgement calls.
- **Limited query diversity.** Short English descriptions of single subjects. No negations, counts, spatial relations, long queries or other languages.
- **Not a held-out benchmark.** The queries were written for these eight images, so the result does not measure generalization.
- **Similarity scores are not probabilities**, and margins are not confidence levels.
- **No queries without an answer.** Every query has at least one relevant image, so nothing is known about scores when there is no match.
- **One model, one machine.** No other checkpoints were compared, and the timings come from a single computer and an eight-image batch.

## 13. Lessons learned

- **A perfect score can mean the test is too easy.** The useful response was to check the evaluation code independently and to describe what the dataset cannot show, not to report the score as an achievement.
- **Verify metrics separately from the model.** Keeping the metric functions free of tensors made it possible to test them with rankings worked out by hand.
- **Exercise the code paths that real data does not reach.** The failure report was only tested because wrong labels were introduced on purpose.
- **Raw CLIP similarities are small and close together.** Correct matches scored between 0.23 and 0.29. Rankings matter; absolute values do not.
- **Multi-answer queries distort the margin.** A small gap between two correct results looks like uncertainty unless it is separated out.
- **Check library return types.** In Transformers 5.x the feature methods return an output object whose `pooler_output` holds the projected embedding. Indexing the result directly, as older examples do, returns per-patch or per-token vectors of the wrong shape without raising an error.
- **Padding needs checking, not assuming.** Batched and individual text embeddings were compared directly before the batched results were trusted.
- **Benchmark numbers vary between runs.** Absolute times shifted by up to about 16% across three runs minutes apart, while the ratio was stable. Warm-up runs, repeated measurements, alternating order and medians all helped.
- **Silent truncation was avoided.** Queries over 77 tokens are rejected with an error so the model never sees a different query from the one typed.

## 14. Potential future experiments

- **A harder dataset.** Several images per category (multiple dogs, multiple beaches) so that queries must discriminate between similar images, using images with licences that allow publication.
- **Queries with no relevant image**, to see whether the top score or margin separates "found" from "not found".
- **More query types:** negation, counting, spatial relations, attributes such as colour, and non-English queries.
- **Prompt wording.** Compare a bare query with a template such as "a photo of ...".
- **Model comparison.** Run the same evaluation with larger CLIP checkpoints.
- **Batch-size scaling.** Time batches of 1, 2, 4, 8, 16 and 32 images to see where the benefit levels off.
- **Thread count and device.** Repeat the benchmark with one CPU thread and with Apple's GPU backend (MPS).
- **Timing breakdown.** Measure preprocessing and the model forward pass separately to see where the batching gain comes from.
- **More labellers.** Have a second person label relevance and measure agreement.
