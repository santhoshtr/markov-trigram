---
title: "From Tokens to Text: A Trigram Markov Model for Malayalam"
date: 2026-01-23
draft: true
tags: ["NLP", "Malayalam", "Language Modeling", "Trigram", "CSF", "Rust"]
categories: ["Natural Language Processing", "Machine Learning"]
series: ["Malayalam Language Modeling"]
toc: true
math: true
description: "Evaluating a Malayalam Unigram tokenizer with a trigram Markov model stored in a Compressed Sparse Fiber (CSF) structure, plus a live generator you can try."
---

In Part 1, I built a Unigram tokenizer tailored for Malayalam. In this post, I evaluate it using a classic statistical language model: trigrams. Along the way, we’ll wrestle with sparsity, pick the right data structure (CSF), and build a generator that you can try online.

Links to the series:
- Part 1: Why Malayalam Needs Better Tokenizers (link when published)
- Part 2 (this post): Trigram Markov modeling

Live demo: https://malgen.thottingal.in/

Source corpus: Swathanthra Malayalam Computing (SMC) corpus — https://gitlab.com/smc/corpus/

Repository: [TODO: https://github.com/santhoshtr/markov-trigram]

## Why Trigrams?

A trigram model estimates the probability of the next token given the previous two tokens:

$$P(w_3 \mid w_1, w_2) = \frac{\operatorname{Count}(w_1, w_2, w_3)}{\operatorname{Count}(w_1, w_2)}$$

For Malayalam, this gives enough context to capture many local syntactic and morphological dependencies, while staying tractable to build and query.

| Model | Context | Data Needed | Accuracy | Memory |
|---|---|---|---|---|
| Unigram | None | ~1K examples | Low | Minimal |
| Bigram | 1 token | ~10K examples | Good | Small |
| **Trigram** | 2 tokens | ~100K examples | **Excellent** | **Moderate** |
| 4-gram | 3 tokens | ~1M examples | Marginal gain | Large |

## From Text To Trigrams (With Our Tokenizer)

Pipeline overview:

1) Train Unigram tokenizer (Part 1) → `data/tokenizer.ml.json`
2) Encode corpus lines into token IDs
3) Extract sliding-window trigrams of IDs
4) Count occurrences

Example:

```
Input:  "... മലയാളം വാചകം ..."
Tokens: [w₁, w₂, w₃, w₄, ...]
Trigrams: (w₁,w₂,w₃), (w₂,w₃,w₄), ...
```

{{< figure src="/images/TODO-trigram-window.png" caption="Placeholder: Sliding window over token IDs to form trigrams." >}}

## The Sparsity Wall

With a 16K-vocabulary tokenizer, the space of possible trigrams is 16K³ ≈ 4 trillion. A dense 3D array is impossible (≈16 TB for counts alone). Real corpora, however, are extremely sparse—only a tiny fraction of possible trigrams ever occur.

| Storage | Dimension | Memory (example) | Notes |
|---|---|---|---|
| Dense 3D array | 3D | ~16 TB | Infeasible |
| HashMap | Flat | 10s–100s MB | Fast, but poor for range-queries |
| CSR (flattened) | 2D | ~GBs | Designed for matrices |
| **CSF** | **3D** | **Compact + fast range** | **Purpose-built for tensors** |

## CSF: Compressed Sparse Fiber

CSF extends the idea of CSR to tensors. For trigrams, it organizes counts in three levels: w₁ → w₂ → w₃. Each level uses pointer arrays to define ranges, and all indices per range are sorted for binary search.

High-level structure:

```
Level 1 (w₁):
  w1_to_idx: HashMap<u32, usize>
  w1_ptr:    Vec<usize>     // ranges into w2_indices

Level 2 (w₂):
  w2_indices: Vec<u32>      // sorted per w₁
  w2_ptr:     Vec<usize>     // ranges into w3_indices

Level 3 (w₃):
  w3_indices: Vec<u32>      // sorted per (w₁,w₂)
  counts:     Vec<u32>

Metadata: vocabulary_size, bigram_totals, unigram_totals
```

Lookup sketch:

```
GetCount(w₁, w₂, w₃):
  w1_idx = w1_to_idx[w₁]
  [w2_start, w2_end) = w1_ptr[w1_idx .. w1_idx+1]
  w₂ position = binary_search(w2_indices[w2_start..w2_end], w₂)
  [w3_start, w3_end) = w2_ptr[w₂ position .. +1]
  w₃ position = binary_search(w3_indices[w3_start..w3_end], w₃)
  return counts[w3_start + w₃ position]
```

{{< figure src="/images/TODO-csf-diagram.png" caption="Placeholder: Three-level CSF layout for (w₁,w₂,w₃) with pointer arrays." >}}

Why CSF here:
- Efficient range queries: given (w₁,w₂), get all w₃ candidates directly
- Cache-friendly, contiguous arrays
- Minimal overhead vs nested maps

## Building The Model (Rust)

The project includes a CLI to build and query the model. It processes the corpus in chunks, parallelizes over lines, and converts a nested map into CSF arrays.

Build the model:

```bash
cargo run --release --bin markov-trigram -- build \
  -d corpus/ \
  -o model.bin \
  -t data/tokenizer.ml.json \
  -m 1024
```

What you’ll see:

```
Building trigram model from corpus directory: corpus/
Using tokenizer: data/tokenizer.ml.json
Found N text file(s) to process
[1/N] Processing: ...
  ✓ Completed: ...
...
Model built with X trigrams
Memory usage: Y MB
Model saved to: model.bin
```

{{< figure src="/images/TODO-build-progress.png" caption="Placeholder: Terminal output showing multi-file progress and final stats." >}}

## Querying and Probabilities

You can query individual trigram counts or probabilities:

```bash
cargo run --release --bin markov-trigram -- query \
  -m model.bin \
  -t data/tokenizer.ml.json \
  --w1 "വാക്ക്1" --w2 "വാക്ക്2" --w3 "വാക്ക്3"
```

Probability uses counts and bigram totals; Laplace smoothing reserves mass for unseen trigrams.

## Generation: Try It Live

The generator samples next tokens proportional to counts for the current (w₁,w₂) context, sliding the window forward each step. You can try it online:

Demo: https://malgen.thottingal.in/

{{< figure src="/images/TODO-demo-screenshot.png" caption="Placeholder: Screenshot of malgen.thottingal.in with a prompt and generated continuation." >}}

CLI generation example:

```bash
cargo run --release --bin markov-trigram -- generate \
  --prompt "ഒരു നല്ല" \
  --max-tokens 50 \
  --seed 42 \
  -m model.bin \
  -t data/tokenizer.ml.json
```

Sample outputs (placeholders; to be replaced with your captured generations):

```
Prompt: "ഒരു നല്ല"
Output: TODO (paste 2–3 diverse examples)

Prompt: "കവിത"
Output: TODO
```

## Performance & Memory (To Be Filled)

Please replace the placeholders below with actual measurements from your build and machine.

| Metric | Value | Notes |
|---|---:|---|
| Vocabulary size | 16,000 (or TODO) | From tokenizer |
| Unique trigrams | TODO | From build log |
| Model size (disk) | TODO MB | `model.bin` |
| Peak build memory | TODO MB | From system monitor |
| Query latency | TODO ms | Average per query |
| Generation speed | TODO tokens/s | With/without seed |

{{< figure src="/images/TODO-performance-chart.png" caption="Placeholder: Simple bar chart of memory and speed vs alternatives (HashMap, CSR, CSF)." >}}

## Data: SMC Corpus Snapshot (Placeholders)

The model was trained on the SMC Malayalam corpus (free-licensed). Please fill in the exact numbers you used:

| Source | Documents | Size | Domain |
|---|---:|---:|---|
| Wikipedia | TODO | TODO | Encyclopedia |
| Sayahna | TODO | TODO | Literature |
| News | TODO | TODO | Current events |
| Other | TODO | TODO | Mixed |
| Total | TODO | TODO | — |

Corpus: https://gitlab.com/smc/corpus/

{{< figure src="/images/TODO-corpus-snapshot.png" caption="Placeholder: Small screenshot of the local corpus directory listing." >}}

## What We Learned

- A good tokenizer dramatically improves efficiency for Malayalam: fewer tokens, better coverage, more coherent downstream modeling.
- Trigrams offer a practical balance of context and tractability for evaluation and generation.
- CSF enables memory- and cache-friendly storage for sparse 3D trigram counts, with fast range queries essential for generation.

## What’s Next

Possible future directions:
- Larger corpora and controlled domain subsets (literature vs news)
- 4-gram backoff models, Kneser-Ney smoothing
- Neural LMs that reuse this tokenizer
- Alignment with ASR/Spell-checking tasks for Malayalam

{{< notice info >}}
Try the demo at https://malgen.thottingal.in/ and the CLI in the repository [TODO: https://github.com/santhoshtr/markov-trigram]. Feedback and corpus contributions are welcome!
{{< /notice >}}
