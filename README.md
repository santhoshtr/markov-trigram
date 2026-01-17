# Markov Trigram Language Model: Algorithms & Data Structures

A high-performance Rust implementation of an n-gram language model using **Compressed Sparse Fiber (CSF)** representation for efficient storage and query of trigram statistics.

**Table of Contents**
- [What is a Trigram Language Model?](#what-is-a-trigram-language-model)
- [Why Trigrams?](#why-trigrams)
- [Tokenization: From Text to Numbers](#tokenization-from-text-to-numbers)
  - [What is Tokenization?](#what-is-tokenization)
  - [Subword Tokenization](#subword-tokenization)
  - [Unigram Algorithm](#unigram-algorithm)
  - [The ml_tokenizer Binary](#the-ml_tokenizer-binary)
  - [Two-Step Workflow](#two-step-workflow)
  - [Vocabulary Size Considerations](#vocabulary-size-considerations)
- [The Sparsity Problem](#the-sparsity-problem)
- [Sparse Tensor Representations](#sparse-tensor-representations)
- [CSF vs CSR: Why We Chose CSF](#csf-vs-csr-why-we-chose-csf)
- [Implementation Details](#implementation-details)
- [Building the Model](#building-the-model)
- [Querying the Model](#querying-the-model)
- [Performance Analysis](#performance-analysis)
- [Usage Examples](#usage-examples)

---

## What is a Trigram Language Model?

### Basic Concept

A **trigram language model** estimates the probability of the next word given two preceding words.
It answers the question: *"What's the probability of word w₃ appearing after the sequence of words w₁ and w₂?"*

Mathematically:
$$P(w_3 | w_1, w_2) = \frac{\text{Count}(w_1, w_2, w_3)}{\text{Count}(w_1, w_2)}$$

Where:
- **Count(w₁, w₂, w₃)** = number of times the trigram appears in training data
- **Count(w₁, w₂)** = number of times the bigram (w₁, w₂) appears

### Concrete Example

Consider the sentence: *"The quick brown fox jumps over the lazy dog"*

Tokenized: `[The, quick, brown, fox, jumps, over, the, lazy, dog]`

*(Note: For simplicity, this example shows word-level tokenization. In practice, the model uses subword tokenization with numeric token IDs. See [Tokenization: From Text to Numbers](#tokenization-from-text-to-numbers) below for details.)*

This sentence contains these trigrams:
- (The, quick, brown) → count = 1
- (quick, brown, fox) → count = 1
- (brown, fox, jumps) → count = 1
- etc.

If our entire corpus contains 100 occurrences of the bigram (The, quick) and 42 of those are followed by "brown", then:
$$P(\text{brown} | \text{The}, \text{quick}) = \frac{42}{100} = 0.42$$

### Why This Matters

Language models are fundamental building blocks for:
- **Text Generation**: Generate plausible continuations of text
- **Speech Recognition**: Disambiguate homophones (hear vs here)
- **Machine Translation**: Rank candidate translations by fluency
- **Spell Checking**: Suggest corrections that maintain grammatical plausibility
- **Information Retrieval**: Score document relevance

---

## Why Trigrams?

Let's compare different n-gram models on the **context-memory spectrum**:

| Aspect | Unigram | Bigram | Trigram | 4-gram | 5-gram |
|--------|---------|--------|---------|--------|--------|
| **Context** | None: P(w₃) | 1 word: P(w₃\|w₂) | 2 words: P(w₃\|w₁,w₂) | 3 words | 4 words |
| **Sparsity** | Very sparse | Sparse | Moderate | Dense | Very dense |
| **Data needed** | ~1K examples | ~10K examples | ~100K examples | ~1M examples | ~10M examples |
| **Model size** | 16K tokens | 256M pairs | 4B+ triples | Impractical | Infeasible |
| **Accuracy** | Poor | Good | Excellent | Very good | Slight improvement |
| **Lookup speed** | O(1) | O(log n) | O(log n) | O(log n) | O(log n) |

### The Trigram Sweet Spot

**Trigrams offer the best balance:**

1. **Sufficient Context**: Two preceding words capture most grammatical and semantic dependencies
   - *"The **quick** brown"* strongly constrains the next word
   - Most English dependencies occur within 2-3 word windows

2. **Manageable Sparsity**: While 4 billion possible trigrams exist, real text exhibits heavy Zipfian distribution
   - A typical corpus contains only 1-10% of possible trigrams
   - Perfect use case for sparse representation

3. **Data Efficiency**: ~100K corpus examples trains adequate trigram models
   - Unigrams need ~1K examples (too simple)
   - 4-grams need ~1M examples (impractical for many languages)

4. **Computational Balance**: Fast to query, reasonable to build
   - Unigrams: Too simplistic, poor quality
   - Bigrams: Useful but less accurate
   - Higher n-grams: Often overfit, require massive corpora

---

## Tokenization: From Text to Numbers

Before building a trigram model, raw text must be converted into numeric token IDs.
This section explains why tokenization is essential, how it works, and how to use the `ml_tokenizer` binary.

### What is Tokenization?

**Tokenization** is the process of converting human-readable text into a sequence of numeric IDs that computers can process efficiently.

**Why it's needed:**
- Computer memory stores numbers, not strings
- Trigrams store u32 integers (4 bytes each), not string references (heap allocations)
- Integer comparisons are O(1) vs string comparisons which are O(n)
- Fixed vocabulary enables sparse storage (CSF structure)

**Visual example:**

```
Input text:      "Hello world"  →  Tokenizer  →  Token IDs: [4521, 892]
(English)

Input text:      "നമസ്കാരം"    →  Tokenizer  →  Token IDs: [8234, 3156, 7891, 2401]
(Malayalam)
```

**Important:** In the trigram formula, w₁, w₂, and w₃ are actually these numeric token IDs:

$$\text{Trigram}(w_1, w_2, w_3) = (142, 1847, 3421) \rightarrow \text{count} = 42$$

The tokenizer maintains a mapping: Token ID ↔ Token String (e.g., 142 ↔ "The")

### Subword Tokenization

The challenge: Can't store every possible word (infinite vocabulary), but need to cover text effectively.

**Solution: Break words into reusable subword pieces**

Instead of word-level tokens:
```
"The quick brown" → ["The", "quick", "brown"]  // Vocabulary size: 100,000+ 😢
```

Use subword tokens (smaller, reusable pieces):
```
"The quick brown" → ["▁The", "▁quick", "▁brown"]  // Vocabulary size: 16,000 ✓
```

**Why subword tokenization is better:**
1. **Finite vocabulary**: 16,000 tokens covers ~99% of any language
2. **Handles rare words**: "supercalifragilistic" decomposes instead of becoming `<unk>`
3. **Language-agnostic**: Same approach works for English, Malayalam, Japanese, etc.
4. **Efficient storage**: Smaller vocabulary = smaller trigram model

**Concrete examples:**

English:
```
"unhappiness"         → ["▁un", "happiness"]
"internationally"     → ["▁inter", "national", "ly"]
"don't"              → ["▁don", "'", "t"]
```

Malayalam (Indic script):
```
"എങ്ങനെയുണ്ട്?" (How are you?)  → ["▁എ", "ങ്ങനെ", "യ", "ുണ്ട്", "?"]
"നമസ്കാരം" (Hello)              → ["▁ന", "മ", "സ്", "കാ", "രം"]
```

**Note on Metaspace:** The `▁` character (U+2581, "Lower One Eighth Block") represents word boundaries.
This allows the tokenizer to distinguish:

- `"hello"` (word by itself) vs
- `hello` (part of "hello world")

### Unigram Algorithm

The `ml_tokenizer` binary uses the **Unigram tokenization algorithm** to learn vocabulary from your corpus.
This is a probabilistic approach used by modern NLP systems (SentencePiece, XLNet, mBART, etc.).

**How it works (high level):**

1. **Initialization**: Start with a large vocabulary of all character n-grams (unicode characters, bigrams, trigrams, etc.)

2. **Probability Assignment**: Assign probability to each token based on corpus frequency:
   $$P(\text{token}) \propto \text{count in corpus}$$

3. **Optimization (EM algorithm)**:
   - Maximize likelihood: $$\mathcal{L} = \sum_{sentence} \log P(\text{sentence})$$
   - Iteratively adjust probabilities
   - Remove lowest-probability tokens
   - After convergence: Keep top K tokens (default: 16,000)

4. **Segmentation (Inference)**: Use **Viterbi dynamic programming** to find best tokenization:
   ```
   sentence = "international"

   Find segmentation maximizing: P(token₁) × P(token₂) × ... × P(tokenₙ)

   Possibilities:
     - ["▁inter", "national", "ly"]     score: P(inter) × P(national) × P(ly)
     - ["▁i", "n", "t", "er", "n", ...] score: P(i) × P(n) × P(t) × ...

   Viterbi chooses: ["▁inter", "national", "ly"]  (highest score)
   ```
   Time complexity: **O(n²)** for string length n

**Why Unigram vs alternatives:**
- **BPE** (Byte Pair Encoding): Deterministic, less flexible
- **WordPiece** (BERT): English-biased, requires language-specific initialization
- **Unigram**: Probabilistic, works for all languages, optimal segmentation

**References:**
- Kudo (2018): "Subword Regularization: Improving Neural Network Translation Models without Parallel Data"
- SentencePiece: https://github.com/google/sentencepiece

### The ml_tokenizer Binary

The `ml_tokenizer` binary trains custom tokenizers for your specific corpus and language.

**Training a tokenizer:**

```bash
cargo run --release --bin ml_tokenizer -- train \
  -f corpus/ \              # Directory with .txt training files
  -v 16000 \                # Vocabulary size (default: 16000)
  -o data/tokenizer.ml.json # Output file
```

This command:
1. Discovers all `.txt` files in `corpus/` directory
2. Runs Unigram algorithm on the combined text
3. Creates vocabulary of 16,000 tokens
4. Saves as HuggingFace-compatible JSON file

**Output file structure** (`data/tokenizer.ml.json`):
```json
{
  "version": "1.0",
  "model": {
    "type": "unigram",
    "vocab": [
      ["▁the", 0],
      ["▁a", 1],
      ["▁and", 2],
      ...
    ]
  },
  "normalizer": {...},
  "pre_tokenizer": {...},
  "post_processor": {...},
  "decoder": {...}
}
```

**Special tokens (always included):**
- `<s>` : Start of sequence (ID 0)
- `</s>` : End of sequence (ID 2)
- `<unk>` : Unknown/out-of-vocabulary words
- `<pad>` : Padding token
- `<mask>` : Masking token (for MLM tasks)

**Testing the tokenizer:**

```bash
# Encode English text
cargo run --release --bin ml_tokenizer -- encode \
  -t data/tokenizer.ml.json "Hello, world!"

# Output:
# Input text: Hello, world!
# Tokens: ["▁Hello", ",", "▁world", "!"]
# Token IDs: [4521, 6, 892, 23]
```

```bash
# Encode Malayalam text
cargo run --release --bin ml_tokenizer -- encode \
  -t data/tokenizer.ml.json "നമസ്കാരം"

# Output:
# Input text: നമസ്കാരം
# Tokens: ["▁ന", "മ", "സ്", "കാ", "രം"]
# Token IDs: [8234, 5123, 7891, 3156, 2401]
```

### Two-Step Workflow

The markov-trigram project has two distinct phases:

**Step 1: Train Tokenizer** (once per language/corpus)
```
Corpus Text Files → ml_tokenizer train → tokenizer.ml.json
                                        (vocabulary mapping)
```

**Step 2: Build Trigram Model** (uses tokenizer from Step 1)
```
Corpus Text Files + tokenizer.ml.json → markov-trigram build → trigram_model.bin
                                                               (CSF structure)
```

**Complete workflow diagram:**

```
corpus/*.txt (raw training text)
    ↓
    ├─→ ml_tokenizer train -f corpus/ -v 16000 -o data/tokenizer.ml.json
    │   (Learn vocabulary: text → token IDs)
    │   └─→ data/tokenizer.ml.json (16K tokens mapping)
    │
    └─→ markov-trigram build -d corpus/ -t data/tokenizer.ml.json -o model.bin
        (Count trigrams: token IDs → CSF structure)
        └─→ trigram_model.bin (compressed sparse model)

Both files needed for inference:
    ├─→ markov-trigram query -m model.bin -t data/tokenizer.ml.json --w1 The --w2 quick --w3 brown
    │   (Query: "The quick brown" → token IDs → CSF lookup)
    │
    └─→ markov-trigram generate -m model.bin -t data/tokenizer.ml.json --prompt "The quick"
        (Generate: prompt → token IDs → sampling → token IDs → decode)
```

**Key points:**
- Tokenizer is **reusable**: Train once, use with any model
- Model building **requires** tokenizer: Can't build without it
- Both files needed for **inference**: Querying and generation
- Vocabulary from tokenizer must match model's vocabulary size

### Vocabulary Size Considerations

Choosing the right vocabulary size is a critical decision:

| Size | Use Case | Pros | Cons |
|------|----------|------|------|
| **1K** | Ultra-low-resource | Very small model | High `<unk>` rate, poor accuracy |
| **4K** | Low-resource languages | Small memory footprint | Limited coverage |
| **8K** | Small projects, prototypes | Manageable size | Some unknowns |
| **16K** | **Recommended default** | Good balance | Moderate sparsity |
| **32K** | Large corpora, high-coverage | Better coverage | Higher sparsity, memory overhead |
| **64K+** | Specialized domains | Excellent coverage | Very sparse data, slow queries |

**Language-specific recommendations:**

**Malayalam & Indic Scripts** (16K-20K):
- Complex script with character combinations
- Subword approach essential
- 16K usually sufficient; 20K for large, diverse corpora

**English** (8K-16K):
- Relatively simple morphology
- 8K covers ~95% of text
- 16K recommended for better rare word handling

**Mixed-language corpora** (20K-32K):
- Combine multiple scripts/languages
- Use larger vocabulary to cover all

**How to choose:**

1. **Start with 16K** (default, works well for most cases)
2. **Check coverage**: Train tokenizer, run on sample text
3. **Count `<unk>` tokens**: If >2% of tokens are `<unk>`, increase vocabulary
4. **Monitor sparsity**: Larger vocabulary = more zeros in CSF structure
5. **Retrain if needed**: Easy to retrain with different vocabulary size

**Checking coverage example:**

```bash
# Train with 16K vocabulary
cargo run --bin ml_tokenizer -- train -f corpus/ -v 16000 -o tok.json

# Test on sample text
cargo run --bin ml_tokenizer -- encode -t tok.json "Your test text here"

# Count <unk> tokens in output
# If ≥2% are <unk>: Consider increasing to 20K or 32K
```

---

## The Sparsity Problem

### Why Dense Storage Fails

Consider building a language model for Malayalam text with a 16,000-word vocabulary.

The naive approach: Store all possible trigrams in a 3D array:

```
vocabulary_size = 16,000
possible_trigrams = 16,000 × 16,000 × 16,000 = 4,096,000,000,000 (4 trillion)
storage_per_count = 4 bytes (32-bit integer)
total_memory = 4 trillion × 4 bytes = 16 terabytes
```

**This is completely infeasible!**

### Reality: Sparse Data

Real corpora are heavily sparse:

- **Sherlock Holmes text corpus** (128K lines):
  - Vocabulary: 16,000 tokens
  - Possible trigrams: 4 trillion
  - Actual trigrams observed: ~10,000-100,000
  - **Sparsity: 99.99999%** (only 1 in 40,000 trigrams appear!)

### The Matrix Analogy

Think of trigrams as a 3D tensor where dimensions are:
- **Row**: First word (w₁)
- **Column**: Second word (w₂)
- **Depth**: Third word (w₃)
- **Value**: Count of this trigram

Visualization (simplified, only showing w₃ dimension):

```
                w₂ indices: [0, 1, 2, 3, ...]
              ╔════════════════════════════╗
w₁=0          ║ . . 1 . . . 2 . . . . . . ║  Most entries are 0!
              ║                            ║
w₁=1          ║ . . . . . . . . 1 . . 1 . ║
              ║                            ║
w₁=2          ║ 1 . . . . . . . . . . . . ║
              ║                            ║
w₁=3          ║ . . . . 3 . . . . . . . . ║
              ║                            ║
...          ║ . . . . . . . . . . . . . ║
              ╚════════════════════════════╝
              (Each cell could contain another vector of w₃ values)
```

**Dense storage wastes space on all those zeros!**

---

## Sparse Tensor Representations

We need a format that:
1. Only stores non-zero entries
2. Supports fast lookup: given (w₁, w₂, w₃), retrieve count
3. Supports candidate retrieval: given (w₁, w₂), get all possible w₃ values
4. Minimizes memory overhead

### CSR: Compressed Sparse Row (2D)

CSR is the standard approach for sparse **2D matrices**. Let's see how it works, then why it's suboptimal for 3D trigrams.

#### CSR Structure for a 2D Matrix

```
Dense matrix (4×5):        CSR representation:
1 0 0 0 2                  values:    [1, 2, 3, 4, 5, 6]
0 3 0 0 0                  col_idx:   [0, 4, 1, 3, 0, 2]
0 0 0 4 5                  row_ptr:   [0, 2, 3, 5, 6]
0 0 6 0 0
```

**CSR Algorithm:**

```
To find matrix[row][col]:
  1. Start = row_ptr[row]
  2. End = row_ptr[row + 1]
  3. Binary search col_idx[Start:End] for col
  4. If found at index i, return values[i]
  5. Else return 0
```

**CSR Benefits:**
- O(log k) lookup where k = non-zeros in row
- Sequential memory access (cache-friendly)
- Small memory overhead (just two integer arrays)

#### Why CSR Fails for Trigrams

To use CSR for trigrams, we'd need to "flatten" the 3D tensor into 2D. Options:

**Option 1: Flatten (w₁,w₂) as row, w₃ as column**
```
rows = vocabulary_size² = 256 million
columns = vocabulary_size = 16,000
```
Problems:
- Row pointer array alone: 256M × 8 bytes = 2GB overhead!
- Most rows are empty (sparse × sparse)
- Can't efficiently answer: "What are all w3 candidates for (w1, w2)?"

**Option 2: Use 2×CSR (nested)**
```
outer CSR: (w₁ × w₂) matrix
inner CSR: (w₂ × w₃) matrix for each w₁
```
Problems:
- Still requires flattening one dimension
- More complex access pattern
- Not designed for 3-level hierarchy

### CSF: Compressed Sparse Fiber (3D)

**CSF naturally extends CSR to handle arbitrary tensor dimensions** by maintaining a recursive pointer structure at each level.

#### CSF Structure for 3D Trigrams

The key insight: **Use a hierarchy of pointers to represent each tensor dimension.**

```
┌─────────────────────────────────────────────────────────┐
│ Level 1: w₁ Dimension                                   │
├─────────────────────────────────────────────────────────┤
│ w1_to_idx:  {0→0, 5→1, 7→2, 15→3, ...}                  │
│ w1_ptr:     [0, 3, 5, 8, 10, ...]                        │
│             └─ Start of w₂ entries for each w₁           │
│
│ Level 2: w₂ Dimension (ranges defined by w1_ptr)
├─────────────────────────────────────────────────────────┤
│ w2_indices: [2, 4, 8, 1, 3, 5, 7, 10, 1, 4, ...]        │
│             └─ w₂ values present with each w₁            │
│
│ w2_ptr:     [0, 2, 5, 7, 10, 14, 18, 22, 26, ...]       │
│             └─ Start of w₃ entries for each (w₁,w₂)      │
│
│ Level 3: w₃ Dimension (ranges defined by w2_ptr)
├─────────────────────────────────────────────────────────┤
│ w3_indices: [1, 5, 2, 7, 9, 1, 2, 5, 7, 3, ...]        │
│             └─ w₃ values present with each (w₁,w₂)       │
│
│ counts:     [1, 3, 2, 4, 1, 1, 5, 2, 1, 3, ...]         │
│             └─ Count for each (w₁,w₂,w₃) triple         │
└─────────────────────────────────────────────────────────┘
```

#### CSF Query Algorithm

Given (w₁, w₂, w₃), find the count:

```
Algorithm: CSF_GetCount(w₁, w₂, w₃)

  // Level 1: Find w₁
  if w₁ ∉ w1_to_idx:
    return 0
  w1_idx ← w1_to_idx[w₁]

  // Get range of w₂ entries for this w₁
  w2_start ← w1_ptr[w1_idx]
  w2_end ← w1_ptr[w1_idx + 1]

  // Level 2: Binary search for w₂
  w2_slice ← w2_indices[w2_start : w2_end]
  w2_rel_idx ← BinarySearch(w2_slice, w₂)
  if w₂ not found:
    return 0
  w2_abs_idx ← w2_start + w2_rel_idx

  // Get range of w₃ entries for this (w₁, w₂)
  w3_start ← w2_ptr[w2_abs_idx]
  w3_end ← w2_ptr[w2_abs_idx + 1]

  // Level 3: Binary search for w₃
  w3_slice ← w3_indices[w3_start : w3_end]
  w3_rel_idx ← BinarySearch(w3_slice, w₃)
  if w₃ not found:
    return 0

  return counts[w3_start + w3_rel_idx]
```

**Time Complexity:** O(log k₁ + log k₂ + log k₃)
- k₁ = number of distinct w₁ values (~tens to thousands)
- k₂ = number of distinct w₂ values per w₁ (~hundreds)
- k₃ = number of distinct w₃ values per (w₁,w₂) (~tens)

**Typical lookup:** O(log 1000 + log 500 + log 50) ≈ O(10 + 9 + 6) ≈ **O(25 operations)**

#### Memory Comparison

For the Sherlock Holmes corpus (16K vocabulary, ~50K trigrams):

```
Dense 3D Array:
  Memory: 16,000 × 16,000 × 16,000 × 4 bytes = 16 TB ❌ INFEASIBLE

CSR (flattened):
  Memory: (256M rows × 8) + (50K × 8) + (50K × 4) ≈ 2GB + overhead ❌ TOO MUCH

CSF (3-level hierarchy):
  w1_to_idx:  2K entries × 16 bytes ≈ 32 KB
  w1_ptr:     1K pointers × 8 bytes ≈ 8 KB
  w2_indices: 50K entries × 4 bytes ≈ 200 KB
  w2_ptr:     50K pointers × 8 bytes ≈ 400 KB
  w3_indices: 50K entries × 4 bytes ≈ 200 KB
  counts:     50K entries × 4 bytes ≈ 200 KB
  bigram_totals: 256M × 4 bytes ≈ 1 GB
  unigram_totals: 16K × 4 bytes ≈ 64 KB
  ───────────────────────────────────────────
  Total: ≈ 1 GB ✓ PRACTICAL
```

---

## CSF vs CSR: Why We Chose CSF

### Feature Comparison Table

| Feature | CSR (2D) | CSF (3D) |
|---------|----------|---------|
| **Natural fit for 3D** | Requires flattening | Native support |
| **Lookup time** | O(log k) per level | O(log k₁ + log k₂ + log k₃) |
| **Memory overhead** | Low (1 pointer array) | Low (2 pointer arrays) |
| **Range queries** | Efficient | Excellent |
| **Sparse × sparse** | Problematic | Excellent |
| **Implementation complexity** | Simple | Moderate |
| **Support for tensors** | No (2D only) | Yes (any dimension) |

### Why CSR Fails for Our Use Case

**Scenario:** Query all possible next words given (w₁, w₂)

```
CSR approach (row = w₁×w₂, col = w₃):
  row_idx = w₁ * vocab_size + w₂
  // This row might be empty! CSR stores no row_ptr entry
  // Must scan sparse matrix to find all columns
  // Very inefficient!

CSF approach:
  w1_idx ← w1_to_idx[w₁]
  w2_start ← w1_ptr[w1_idx]
  w2_end ← w1_ptr[w1_idx + 1]
  w2_abs_idx ← BinarySearch(w2_indices[w2_start:w2_end], w₂)
  w3_start ← w2_ptr[w2_abs_idx]
  w3_end ← w2_ptr[w2_abs_idx + 1]
  // All w₃ candidates are in w3_indices[w3_start:w3_end] ✓
  // Direct sequential access, cache-friendly!
```

**CSF is purpose-built for multi-dimensional sparse data.**

---

## Implementation Details

### Data Structure Definition

```rust
pub struct SparseTrigram {
    // HashMap for fast w₁ → index mapping
    pub w1_to_idx: HashMap<u32, usize>,

    // Level 1 pointers: define ranges in w2_indices
    pub w1_ptr: Vec<usize>,

    // Level 2: actual w₂ values (sorted per w₁)
    pub w2_indices: Vec<u32>,

    // Level 2 pointers: define ranges in w3_indices
    pub w2_ptr: Vec<usize>,

    // Level 3: actual w₃ values (sorted per (w₁,w₂))
    pub w3_indices: Vec<u32>,

    // Level 3 values: trigram counts
    pub counts: Vec<u32>,

    // Metadata
    pub vocabulary_size: usize,
    pub total_trigrams: usize,

    // For smoothing (store cumulative sums)
    pub bigram_totals: Vec<u32>,    // Sum of counts for each (w₁,w₂)
    pub unigram_totals: Vec<u32>,   // Sum of counts for each w₁
}
```

### Invariant Properties

The CSF structure maintains these invariants:

1. **Sorted at every level:**
   ```
   For each w₁: w2_indices[start:end] is sorted
   For each (w₁,w₂): w3_indices[start:end] is sorted
   ```

2. **No duplicates:**
   ```
   Each (w₁,w₂,w₃) triple appears exactly once
   ```

3. **Pointer validity:**
   ```
   w1_ptr.len() = number of distinct w₁ + 1
   w2_ptr.len() = w2_indices.len() + 1
   w1_ptr[i] always < w1_ptr[i+1]  (no empty ranges)
   ```

---

## Building the Model

### Tokenization Step (Preprocessing)

Before counting trigrams, each line of raw text is converted to token IDs using the tokenizer:

```
line = "The quick brown fox"
token_ids = tokenizer.encode(line)
// Result: [142, 1847, 3421, 9876]
```

These u32 integers are what the model actually stores in the CSF structure. The tokenizer file maintains the mapping (e.g., 142 ↔ "The") for display and inference.

**Processing flow:**
1. Load tokenizer from file (e.g., `data/tokenizer.ml.json`)
2. Read each line from corpus files
3. Normalize text (lowercase, unicode normalization, etc.)
4. Tokenize: text → token IDs using Unigram model
5. Extract trigrams from token ID sequences
6. Count occurrences in CSF structure

### Phase 1: Collection (In-Memory HashMap)

During corpus processing, we use a **nested HashMap** for flexibility:

```rust
// Temporary structure: HashMap<w₁, BTreeMap<w₂, HashMap<w₃, count>>>
data: HashMap<u32, BTreeMap<u32, HashMap<u32, u32>>>
```

**Why this structure?**

- **HashMap for w₁**: Fast insertion/lookup of w₁ entries
- **BTreeMap for w₂**: Automatic sorting (needed for CSF)
- **HashMap for w₃**: Fast count updates

**Example:**
```
data = {
  0: {  // w₁ = 0
    2: {4: 3, 7: 1},      // (0,2,4):3  (0,2,7):1
    5: {1: 2, 9: 1}       // (0,5,1):2  (0,5,9):1
  },
  5: {  // w₁ = 5
    0: {2: 1, 3: 1},      // (5,0,2):1  (5,0,3):1
    4: {1: 5}             // (5,4,1):5
  },
  ...
}
```

### Phase 2: Parallel Processing

The builder reads corpus files in **10,000-line chunks** and processes each chunk in parallel:

```rust
Algorithm: ProcessCorpus(file_path)

  const CHUNK_SIZE = 10,000
  buffer ← empty list
  tokenizer ← load tokenizer

  for each line in file:
    buffer.append(line)

    if buffer.size() >= CHUNK_SIZE:
      // Process chunk in parallel
      local_trigrams ← ProcessLinesParallel(buffer)
      merge local_trigrams into data
      buffer.clear()

  if buffer not empty:
    local_trigrams ← ProcessLinesParallel(buffer)
    merge local_trigrams into data
```

```rust
Algorithm: ProcessLinesParallel(lines) → Vec<HashMap<(w₁,w₂,w₃), count>>

  // Rayon parallelizes over lines
  return lines.par_iter().map(|line| {
    local_map ← empty HashMap

    // Tokenize entire line
    tokens ← tokenizer.encode(line)

    // Extract all trigrams from tokens
    for i = 2 to tokens.len():
      (w₁, w₂, w₃) ← (tokens[i-2], tokens[i-1], tokens[i])
      local_map[(w₁, w₂, w₃)] += 1

    return local_map
  }).collect()
```

**Complexity:**
- Lines are processed in parallel (speedup: ~N cores)
- Each tokenization: O(line_length × tokenizer_complexity)
- Trigram extraction: O(tokens)
- Local merge: O(trigrams_in_chunk)

### Phase 3: CSF Conversion

Final step: convert HashMap structure to CSF arrays.

```rust
Algorithm: BuildCSF(data)

  // Step 1: Sort w₁ keys deterministically
  w1_keys ← sort(data.keys())

  // Step 2: Create w₁ mapping
  w1_to_idx ← {}
  for (idx, w₁) in enumerate(w1_keys):
    w1_to_idx[w₁] ← idx

  // Step 3: Build three-level structure
  w1_ptr ← [0]
  w2_indices ← []
  w2_ptr ← []
  w3_indices ← []
  counts ← []
  bigram_totals ← [0 for _ in range(vocab_size²)]
  unigram_totals ← [0 for _ in range(vocab_size)]

  for w₁ in w1_keys:
    w2_map ← data[w₁]
    w2_keys ← sort(w2_map.keys())

    for w₂ in w2_keys:
      // Record this w₂
      w2_indices.append(w₂)
      w2_ptr.append(w3_indices.len())

      // Process all w₃ for this (w₁, w₂)
      w3_map ← w2_map[w₂]
      w3_entries ← sort([(w₃, count) for (w₃, count) in w3_map.items()])

      for (w₃, count) in w3_entries:
        w3_indices.append(w₃)
        counts.append(count)

        // Update totals
        bigram_totals[w₁ * vocab_size + w₂] += count
        unigram_totals[w₁] += count

    // Record end of w₂ entries for this w₁
    w1_ptr.append(w2_indices.len())

  // Final pointer for w2_ptr
  w2_ptr.append(w3_indices.len())

  return SparseTrigram {
    w1_to_idx, w1_ptr, w2_indices, w2_ptr,
    w3_indices, counts, bigram_totals, unigram_totals
  }
```

**Complexity:** O(T log T) where T = total trigrams (due to sorting)

**Memory at conversion time:**
- HashMap structure: ~1-2 GB
- Final CSF structure: ~1 GB
- Total during build: ~2-3 GB (for typical corpora)

---

## Querying the Model

### Single Trigram Lookup

```rust
Algorithm: GetCount(w₁, w₂, w₃)

  // Constant-time HashMap lookup
  if w₁ ∉ w1_to_idx:
    return 0
  w1_idx ← w1_to_idx[w₁]

  // Get w₂ range for this w₁
  w2_start ← w1_ptr[w1_idx]
  w2_end ← w1_ptr[w1_idx + 1]

  if w2_start == w2_end:  // No w₂ entries for this w₁
    return 0

  // Binary search for w₂
  w2_slice ← w2_indices[w2_start : w2_end]
  w2_rel_idx ← BinarySearch(w2_slice, w₂)
  if not found:
    return 0

  w2_abs_idx ← w2_start + w2_rel_idx

  // Get w₃ range for this (w₁, w₂)
  w3_start ← w2_ptr[w2_abs_idx]
  w3_end ← w2_ptr[w2_abs_idx + 1]

  if w3_start == w3_end:  // No w₃ entries for this (w₁, w₂)
    return 0

  // Binary search for w₃
  w3_slice ← w3_indices[w3_start : w3_end]
  w3_rel_idx ← BinarySearch(w3_slice, w₃)
  if not found:
    return 0

  return counts[w3_start + w3_rel_idx]
```

**Time Complexity:** O(log k₁ + log k₂ + log k₃)
**Space Complexity:** O(1)

### Range Query: Get All w₃ Candidates

```rust
Algorithm: GetW3Candidates(w₁, w₂) → Vec<(w₃, count)>

  if w₁ ∉ w1_to_idx:
    return []

  w1_idx ← w1_to_idx[w₁]
  w2_start ← w1_ptr[w1_idx]
  w2_end ← w1_ptr[w1_idx + 1]

  if w2_start == w2_end:
    return []

  // Binary search for w₂
  w2_slice ← w2_indices[w2_start : w2_end]
  w2_rel_idx ← BinarySearch(w2_slice, w₂)
  if not found:
    return []

  w2_abs_idx ← w2_start + w2_rel_idx

  // Get ALL w₃ for this (w₁, w₂) - direct access!
  w3_start ← w2_ptr[w2_abs_idx]
  w3_end ← w2_ptr[w2_abs_idx + 1]

  results ← []
  for i in range(w3_start, w3_end):
    results.append((w3_indices[i], counts[i]))

  return results
```

**Time Complexity:** O(log k₁ + log k₂ + k₃) where k₃ = candidates
**Space Complexity:** O(k₃) for results

### Probability Calculation

```rust
Algorithm: Probability(w₁, w₂, w₃, smoothing)

  count ← GetCount(w₁, w₂, w₃)

  if count > 0:
    // Maximum likelihood estimation
    bigram_total ← bigram_totals[w₁ * vocab_size + w₂]
    if bigram_total > 0:
      return count / bigram_total

  // Laplace smoothing if count is 0
  if smoothing enabled:
    unigram_total ← unigram_totals[w₁]
    if unigram_total > 0:
      // P(w₃ | w₁, w₂) = α / (unigram_total + α * vocab_size)
      // where α is smoothing parameter
      return α / (unigram_total + α * vocab_size)

  return 0
```

**Why Laplace Smoothing?**

Real text introduces novel trigrams never seen in training:
- Unsmoothed: P(novel | context) = 0 (impossible!)
- Laplace: "Pretend each unseen trigram occurred once more"
- Effect: Reserves small probability mass for unknown trigrams

---

## Performance Analysis

### Lookup Performance

**Comparison: HashMap vs CSF**

```
Operation: Query 100,000 random trigrams

HashMap (collision cost ~1.05 at load factor 0.75):
  Time: 100K × 1.05 = ~105K operations
  Memory: 50K entries × 16 bytes = 800 KB

CSF:
  Time: 100K × (10 + 9 + 6) = ~2.5M operations
  Memory: Multiple arrays, ~500 KB

Verdict: HashMap is faster! But...
```

**However, consider the complete picture:**

```
Total model memory (Sherlock Holmes, 16K vocab):
  Naive HashMap:
    - All 50K trigrams: 50K × 16B = 800 KB
    - w₁, w₂, w₃ overhead: ~400 KB
    - Total: ~1.2 MB (just for lookup)

  CSF:
    - Pointer structures: ~600 KB
    - Dense arrays: ~400 KB
    - Total: ~1 GB (includes metadata for generation)

CSF wins because:
  1. Includes bigram/unigram totals (needed for probability)
  2. Enables efficient range queries (all w₃ candidates)
  3. Better cache locality (sequential array access)
  4. Scales better with larger corpora
```

### Memory Scaling

For vocabulary size V and T unique trigrams:

```
Dense 3D Array:
  Memory: O(V³) = catastrophic!
  Example (V=16K): 16TB

HashMap approach:
  Memory: O(T × k) where k = overhead per entry
  Example (T=50K, k=16B): 800 KB

CSF:
  Memory: O(T + vocab metadata)
  Example (T=50K): 1 GB (includes totals arrays)
```

### Generation Performance

```
Algorithm: Generate(prompt, max_tokens, seed)

  // Initial context
  tokens ← Tokenize(prompt)
  w1, w2 ← tokens[-2], tokens[-1]  // Last two tokens

  for i in range(max_tokens):
    // Get all possible next words
    candidates ← GetW3Candidates(w1, w2)

    if candidates is empty:
      // Backoff: sample from vocabulary
      w3 ← RandomToken()
    else:
      // Sample proportional to counts (roulette wheel)
      total ← sum(counts in candidates)
      r ← RandomInt(0, total)
      w3 ← SelectFromCandidates(candidates, r)

    tokens.append(w3)
    w1, w2 ← w2, w3  // Slide window

  return Detokenize(tokens)
```

**Complexity per token:** O(log k₁ + log k₂ + k₃ + selection)

**Typical runtime:** ~1-2ms per token on modern CPUs

---

## Usage Examples

### Prerequisites: Training a Tokenizer

Before building a trigram model, you must first train a tokenizer from your corpus. The tokenizer creates the vocabulary that the model will use.

**Step 1: Train the tokenizer**

```bash
# Train tokenizer from corpus directory
cargo run --release --bin ml_tokenizer -- train \
  -f corpus/ \                      # Directory with .txt files
  -v 16000 \                        # Vocabulary size (default)
  -o data/tokenizer.ml.json         # Output file

# Or using Makefile
make train-tokenizer
```

**What this does:**
1. Discovers all `.txt` files in `corpus/` directory
2. Runs Unigram tokenizer training algorithm
3. Creates vocabulary of 16,000 tokens
4. Saves as `data/tokenizer.ml.json` (HuggingFace format)

**Output example:**
```
Found 42 text files:
  corpus/book1.txt
  corpus/book2.txt
  ...

Training tokenizer...
[████████████████████] 100%

Tokenizer saved to: data/tokenizer.ml.json

Test text: Hello, world!
Tokens: ["▁Hello", ",", "▁world", "!"]
Token IDs: [4521, 6, 892, 23]
```

**Step 2: Verify tokenizer works**

```bash
# Encode test text
cargo run --release --bin ml_tokenizer -- encode \
  -t data/tokenizer.ml.json "Test your text here"

# Output should show tokens and IDs without errors
```

**Now you can build the trigram model** using this tokenizer (see below).

### Building a Model

```bash
# Using CLI
cargo run --release --bin markov-trigram -- build \
  -d corpus/ \
  -o model.bin \
  -t data/tokenizer.ml.json \
  -m 1024  # 1GB max memory

# Using Makefile
make build-model CORPUS_DIR=corpus/
```

**Output:**
```
Building trigram model from corpus directory: corpus/
Using tokenizer: data/tokenizer.ml.json

Found 42 text file(s) to process

[1/42] Processing: corpus/book1.txt
  ✓ Completed: corpus/book1.txt

[2/42] Processing: corpus/book2.txt
  ✓ Completed: corpus/book2.txt

...

Processing complete:
  ✓ Successfully processed: 42 file(s)

Building final model from 42 successful file(s)...
Model built with 2,847,392 trigrams
Memory usage: 1,247.50 MB
Model saved to: model.bin
```

### Querying Probabilities

```bash
# Query probability: P(was | Sherlock, Holmes)
cargo run --release --bin markov-trigram -- query \
  -m model.bin \
  -t data/tokenizer.ml.json \
  --w1 "Sherlock" \
  --w2 "Holmes" \
  --w3 "was"

# Output:
# P(was | Sherlock, Holmes) = 0.15
# Count: 12
```

### Text Generation

```bash
# Generate 50 tokens from prompt
cargo run --release --bin markov-trigram -- generate \
  --prompt "The detective" \
  --max-tokens 50 \
  --seed 42 \
  -m model.bin \
  -t data/tokenizer.ml.json

# Output (deterministic with seed):
# The detective was a man of considerable experience.
# He had solved many cases before, and his methods were
# always logical and thorough. Watson had learned to trust
# his instincts implicitly. The game was afoot.

# Same command without seed = different output each time
```

### Practical Application: Speech Recognition

```
User says: "I want to hear a song"
Speech recognizer produces: {"I", "want", "to", "here"/"hear", "a", "song"}

Ambiguity: Did user say "here" or "hear"?
  P(a | to, here) = 0.0001  (rare)
  P(a | to, hear) = 0.0512  (common in music context)

Decision: User said "hear" ✓ (CSF enables fast evaluation of both)
```

---

## Conclusion

The **Compressed Sparse Fiber (CSF)** representation is ideal for n-gram language models because:

1. **Naturally handles sparse 3D tensors** without flattening
2. **Efficient lookup** with O(log k₁ + log k₂ + log k₃) complexity
3. **Excellent range queries** for candidate selection
4. **Minimal memory overhead** compared to dense alternatives
5. **Cache-friendly** sequential access patterns
6. **Scales to real corpora** (billions of n-grams on commodity hardware)

The trigram model itself represents an excellent **practical compromise** between:
- **Unigrams**: Too simple, poor quality
- **Bigrams**: Good baseline, but limited context
- **Trigrams**: Excellent accuracy with reasonable memory footprint
- **4+ grams**: Diminishing returns, sparse data, large models

For most NLP applications with English-scale vocabularies (10K-100K tokens) and typical corpora (10M-100M words), **trigram models with CSF storage deliver the best balance of speed, memory, and accuracy.**

---

## References & Further Reading

### Sparse Tensor Fundamentals
- Kolda, T. G., & Bader, B. W. (2009). **Tensor Decompositions and Applications**. SIAM Review, 51(3), 455-500.
- Li, Y., Tarlow, D., Brockschmidt, M., & Zemel, R. (2016). **Gated Graph Sequence Neural Networks**. ICLR 2016.

### CSR/CSF Data Structures
- Saad, Y. (2003). **Iterative Methods for Sparse Linear Systems** (2nd ed.). SIAM.
- Smith, S., Ravindran, N., Sidiropoulos, N. D., & Karypis, G. (2017). **SPLATT: Efficient and Cache-Friendly Sparse Tensor Multiplication**.

### N-gram Language Models
- Chen, S. F., & Goodman, J. (1998). **An Empirical Study of Smoothing Techniques for Language Modeling**. Tech Report, CMU.
- Kneser, K., & Ney, H. (1995). **Improved Backing-Off for M-gram Language Modeling**. ICASSP 1995.

### NLP Fundamentals
- Jurafsky, D., & Martin, J. H. (2021). **Speech and Language Processing** (3rd ed., draft). Available online.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). **Deep Learning**. MIT Press. (Chapter 12: Sequence Modeling)

---

