# Malayalam Markov Trigram

A Rust implementation of a trigram language model using Compressed Sparse Fiber (CSF) representation for efficient storage and retrieval.

## Quick Start

```bash
# Build model from corpus
cargo run -- build -d corpus/ -o model.bin

# Generate text
cargo run -- generate --prompt "മലയാളം" -m model.bin

# Query probability
cargo run -- query -m model.bin --w1 മലയാളം --w2 ഭാഷ --w3 ആണ്
```

For detailed explanation of the algorithms and data structures, see: https://thottingal.in/blog/2026/02/28/malayalam-markov-chain/
