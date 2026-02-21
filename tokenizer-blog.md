---
title: "Why Malayalam Needs Better Tokenizers: Building a Custom Unigram Model"
date: 2026-01-23
draft: true
tags: ["NLP", "Malayalam", "Tokenization", "Unigram", "Machine Learning", "Rust"]
categories: ["Natural Language Processing", "Machine Learning"]
series: ["Malayalam Language Modeling"]
toc: true
math: true
description: "Most LLMs rely on BPE tokenizers that work poorly for Malayalam. This post explains why, and walks through building a custom Unigram tokenizer and preparing a high-quality vocabulary from the SMC corpus."
---

In modern NLP, tokenization is the first, decisive cut. For Malayalam, that cut has often been in the wrong place.

Today, most large language models rely on BPE (Byte-Pair Encoding) or close variants. Those choices made sense for English and similar scripts. For Malayalam—rich morphology, complex orthography, conjuncts—BPE tends to fragment words, inflate token counts, and erode model quality.

This post explains why Malayalam needs a different approach and how I built a custom Unigram tokenizer. In Part 2, we will evaluate it with a trigram Markov model and a live text generator.



References and resources along the way:
- SMC Malayalam Corpus: https://gitlab.com/smc/corpus/
- Live generator (used in Part 2): https://malgen.thottingal.in/
- SentencePiece (Unigram reference implementation): https://github.com/google/sentencepiece
- BPE overview: https://en.wikipedia.org/wiki/Byte_pair_encoding
- Malayalam Unicode chart: https://unicode.org/charts/PDF/U0D00.pdf

## The Problem With BPE for Malayalam

BPE merges frequent byte or character pairs to form tokens. It works well for languages with relatively simple orthography and morphology. Malayalam challenges these assumptions:

- Conjuncts and combining marks create multi-codepoint graphemes
- Productive morphology leads to long surface forms
- Sandhi/coalescence and orthographic conventions complicate word boundaries

The net result: BPE often splits Malayalam words into many short pieces (or even bytes), increasing token counts and harming downstream models.

{{< figure src="/images/TODO-bpe-vs-unigram-malayalam.png" caption="Placeholder: Side-by-side tokenization of a Malayalam sentence with BPE vs Unigram (lower token count, more meaningful pieces)." >}}

Table: Script/Tokenizer Fit

| Aspect | English (Latin) | Malayalam |
|---|---|---|
| Script complexity | Low (alphabetic) | High (combining signs, conjuncts) |
| Morphology | Mostly analytic | Agglutinative/productive |
| Average surface length | ~5 chars/word | 8–12+ chars/word |
| BPE segmentation | Efficient | Over-fragmentation common |

Example symptom: token count inflation

```
English: "hello world"  → ~2–3 tokens
Malayalam: "നമസ്കാരം ലോകമേ" → 8–12 tokens (BPE), 3–6 tokens (Unigram)
```

{{< notice tip >}}
Better segmentation reduces context waste. If your model has a 512-token window, fewer tokens per sentence means more actual text fits, improving coherence and utility.
{{< /notice >}}

## Why Unigram Tokenization

Unigram is a probabilistic tokenization algorithm. Instead of greedily merging pairs (BPE) or relying on language-specific heuristics (WordPiece), it learns a vocabulary of subword pieces and assigns probabilities to each. At inference, it chooses the segmentation that maximizes sentence likelihood.

Advantages for Malayalam:
- Language-agnostic, script-agnostic
- Learns units that align with grapheme clusters and morphemes
- Robust to rare words; avoids `<unk>` by decomposing
- Tends to minimize token count while staying meaningful

Further reading:
- Kudo (2018), Subword Regularization: https://arxiv.org/abs/1804.10959
- SentencePiece docs: https://github.com/google/sentencepiece

### How Unigram Works (Intuition)

1) Start with a large seed vocabulary of character n-grams.
2) Estimate token probabilities from corpus statistics.
3) Optimize via EM; prune low-probability pieces.
4) At inference, use dynamic programming (Viterbi) to find the best segmentation.

Segmentation example:

```
"അങ്ങനെയുണ്ട്" → ["▁അ", "ങ്ങനെ", "യ", "ുണ്ട്"]
```

Contrast this with byte-level or BPE segmentations that may explode token counts for the same word.

{{< figure src="/images/TODO-unigram-viterbi-illustration.png" caption="Placeholder: Unigram segmentation via Viterbi. Show multiple possible paths and the highest-likelihood path." >}}

### Algorithm Comparison

| Algorithm | Type | Strengths | Weaknesses | Used in |
|---|---|---|---|---|
| BPE | Deterministic merges | Fast, simple, strong for Latin scripts | Over-splits complex scripts | GPT-2/3 family, RoBERTa |
| WordPiece | Greedy | Stable vocabularies | Initialization bias, English-centric | BERT |
| Unigram | Probabilistic | Script-agnostic, optimal segmentation | Training is heavier | T5, XLM-R, SentencePiece |

## Building a Malayalam Tokenizer (ml_tokenizer)

I implemented a fast Unigram tokenizer training and inference binary in Rust. It discovers corpus files, trains a vocabulary, saves a HuggingFace-compatible JSON, and can encode text for inspection.

Repository: [TODO: https://github.com/santhoshtr/markov-trigram]

### Train the Tokenizer

```bash
cargo run --release --bin ml_tokenizer -- train \
  -f corpus/ \            # Directory with .txt files
  -v 16000 \              # Vocabulary size (default: 16000)
  -o data/tokenizer.ml.json
```

What it does:
- Recursively finds `.txt` files under `corpus/`
- Trains a Unigram vocabulary
- Writes `data/tokenizer.ml.json` in HuggingFace format

{{< figure src="/images/TODO-tokenizer-training-progress.png" caption="Placeholder: Terminal snapshot showing training progress and final vocabulary size." >}}

### Inspect the Output

```bash
cargo run --release --bin ml_tokenizer -- encode \
  -t data/tokenizer.ml.json "നമസ്കാരം"
```

Expected output (example):

```
Input text: നമസ്കാരം
Tokens: ["▁ന", "മ", "സ്", "കാ", "രം"]
Token IDs: [8234, 5123, 7891, 3156, 2401]
```

{{< figure src="/images/TODO-tokenizer-json-snippet.png" caption="Placeholder: JSON snippet of a few learned tokens and IDs." >}}

### Vocabulary Size: How Big Is Big Enough?

Choosing the vocabulary is a trade-off: larger vocabularies improve coverage but increase sparsity and memory. Start with 16K and tune according to coverage on your target text.

| Vocab Size | Use Case | Pros | Cons |
|---|---|---|---|
| 8K | Prototypes, small corpora | Small, fast | More `<unk>`, longer sequences |
| 16K | Recommended default | Good balance | Moderate sparsity |
| 20K | Large, diverse Malayalam | Better coverage | Larger model, more sparsity |
| 32K | Mixed-language corpora | Broad coverage | Higher memory, slower |

Heuristic:
1) Train at 16K
2) Encode a validation sample
3) If `<unk>` > 2% or tokens/word is high, increase to 20K

```bash
# Example workflow
make train-tokenizer VOCAB_SIZE=16000
cargo run --release --bin ml_tokenizer -- encode -t data/tokenizer.ml.json "സാമ്പിൾ വാചകം"
```

## Training Data: SMC Malayalam Corpus

I trained the tokenizer on the Swathanthra Malayalam Computing (SMC) free-licensed corpus, which aggregates Malayalam text from Wikipedia, the Sayahna project, newspapers, and more.

- Corpus home: https://gitlab.com/smc/corpus/
- Organization: https://smc.org.in/

Corpus snapshot used (placeholders; will update with exact numbers):

| Source | Documents | Size | Notes |
|---|---|---|---|
| Wikipedia | TODO | TODO | Encyclopedic |
| Sayahna | TODO | TODO | Literature |
| News | TODO | TODO | Current events |
| Other | TODO | TODO | Mixed |
| Total | TODO | TODO | — |

{{< figure src="/images/TODO-corpus-structure.png" caption="Placeholder: Directory structure of the local corpus used for training." >}}

## Validation: Does It Actually Help?

Early signs are positive: fewer tokens per word, fewer `<unk>`, and segmentations that align better with Malayalam morphology. For a full evaluation, I measure:

- Coverage: percent of text representable without `<unk>`
- Tokens per word: lower is better (for fixed context windows)
- Stability across genres: news, literature, Wikipedia

Placeholders for results (to be filled with your measurements):

| Metric | Baseline (BPE) | Unigram (this work) |
|---|---:|---:|
| Coverage (%) | TODO | TODO |
| Avg tokens/word | TODO | TODO |
| `<unk>` rate (%) | TODO | TODO |

{{< figure src="/images/TODO-tokenization-comparison-chart.png" caption="Placeholder: Chart comparing BPE vs Unigram on coverage and tokens/word for a held-out set." >}}

{{< notice info >}}
If you want to reproduce this work, clone the repo [TODO: https://github.com/santhoshtr/markov-trigram], train with `make train-tokenizer`, and try encoding your own sentences.
{{< /notice >}}
