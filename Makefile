# ============================================================================
# Markov Trigram Project Makefile
# ============================================================================

# Configuration Variables
# ============================================================================
CORPUS_DIR     := ~/data/corpus
DATA_DIR       := data
MODEL_FILE     := trigram_model.bin
TOKENIZER_FILE := $(DATA_DIR)/tokenizer.ml.json
MAX_MEMORY     := 1024
VOCAB_SIZE     := 16000

# Cargo commands
CARGO          := cargo
CARGO_BUILD    := $(CARGO) build --release
CARGO_RUN      := $(CARGO) run --release --bin markov-trigram --
CARGO_TOK      := $(CARGO) run --release --bin ml_unigram_tokenizer --
CARGO_BPE_TOK  := $(CARGO) run --release --bin ml_bpe_tokenizer --
CARGO_WEB      := $(CARGO) run --release --bin markov-web --

# Colors for output
COLOR_RESET    := \033[0m
COLOR_BOLD     := \033[1m
COLOR_GREEN    := \033[32m
COLOR_YELLOW   := \033[33m
COLOR_BLUE     := \033[34m

# ============================================================================
# Main Targets
# ============================================================================

.PHONY: all help build test clean serve rebuild demo info check-model train-tokenizer train-bpe-tokenizer compare-tokenizers

# Default target
all: help

# Show help message
help:
	@echo ""
	@echo "$(COLOR_BOLD)Markov Trigram Project - Makefile$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Available targets:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)help$(COLOR_RESET)                 - Show this help message"
	@echo "  $(COLOR_GREEN)build-model$(COLOR_RESET)          - Build trigram model from all corpus files"
	@echo "  $(COLOR_GREEN)train-tokenizer$(COLOR_RESET)      - Train unigram tokenizer from corpus"
	@echo "  $(COLOR_GREEN)train-bpe-tokenizer$(COLOR_RESET)  - Train BPE tokenizer from corpus"
	@echo "  $(COLOR_GREEN)compare-tokenizers$(COLOR_RESET)   - Compare tokenizer quality (intrinsic metrics)"
	@echo "  $(COLOR_GREEN)generate$(COLOR_RESET)             - Generate text from a prompt (use PROMPT variable)"
	@echo "  $(COLOR_GREEN)query$(COLOR_RESET)                - Query trigram probability (use W1, W2, W3 variables)"
	@echo "  $(COLOR_GREEN)serve$(COLOR_RESET)                - Start web server (port 3000)"
	@echo "  $(COLOR_GREEN)test$(COLOR_RESET)                 - Run Rust tests"
	@echo "  $(COLOR_GREEN)build$(COLOR_RESET)                - Build Rust binaries"
	@echo "  $(COLOR_GREEN)format$(COLOR_RESET)               - Format Rust code with cargo fmt"
	@echo "  $(COLOR_GREEN)lint$(COLOR_RESET)                 - Run clippy linter"
	@echo "  $(COLOR_GREEN)clean$(COLOR_RESET)                - Clean build artifacts and generated files"
	@echo "  $(COLOR_GREEN)clean-all$(COLOR_RESET)            - Deep clean including tokenizer"
	@echo "  $(COLOR_GREEN)info$(COLOR_RESET)                 - Show configuration info"
	@echo "  $(COLOR_GREEN)demo$(COLOR_RESET)                 - Quick demo with Sherlock Holmes prompt"
	@echo ""
	@echo "$(COLOR_BOLD)Examples:$(COLOR_RESET)"
	@echo "  make build-model"
	@echo "  make train-tokenizer VOCAB_SIZE=20000"
	@echo "  make train-bpe-tokenizer VOCAB_SIZE=16000"
	@echo "  make compare-tokenizers"
	@echo "  make generate PROMPT=\"Sherlock Holmes\" MAX_TOKENS=100"
	@echo "  make generate PROMPT=\"Watson said\" SEED=42"
	@echo "  make query W1=Sherlock W2=Holmes W3=was"

# ============================================================================
# Model Building
# ============================================================================

build-model: build-rust
	@echo "$(COLOR_BOLD)Building trigram model...$(COLOR_RESET)"
	@echo "Corpus directory: $(CORPUS_DIR)"
	@echo ""
	$(CARGO_RUN) build -d $(CORPUS_DIR) -o $(MODEL_FILE) -t $(TOKENIZER_FILE) -m $(MAX_MEMORY)
	@echo ""
	@echo "$(COLOR_GREEN)✓ Model built successfully: $(MODEL_FILE)$(COLOR_RESET)"

# ============================================================================
# Tokenizer Training
# ============================================================================

train-tokenizer:
	@echo "$(COLOR_BOLD)Training Unigram tokenizer...$(COLOR_RESET)"
	@echo "Corpus directory: $(CORPUS_DIR)"
	@echo "Vocabulary size: $(VOCAB_SIZE)"
	@echo "Output: $(TOKENIZER_FILE)"
	@mkdir -p $(DATA_DIR)
	$(CARGO_TOK) train \
		-f $(CORPUS_DIR) \
		-v $(VOCAB_SIZE) \
		-o $(TOKENIZER_FILE)
	@echo ""
	@echo "$(COLOR_GREEN)✓ Unigram tokenizer trained: $(TOKENIZER_FILE)$(COLOR_RESET)"

# Training BPE tokenizer
TOKENIZER_BPE_FILE := $(DATA_DIR)/tokenizer.ml.bpe.json

train-bpe-tokenizer:
	@echo "$(COLOR_BOLD)Training BPE tokenizer...$(COLOR_RESET)"
	@echo "Corpus directory: $(CORPUS_DIR)"
	@echo "Vocabulary size: $(VOCAB_SIZE)"
	@echo "Output: $(TOKENIZER_BPE_FILE)"
	@mkdir -p $(DATA_DIR)
	$(CARGO_BPE_TOK) train \
		-f $(CORPUS_DIR) \
		-v $(VOCAB_SIZE) \
		-m 2 \
		-o $(TOKENIZER_BPE_FILE)
	@echo ""
	@echo "$(COLOR_GREEN)✓ BPE tokenizer trained: $(TOKENIZER_BPE_FILE)$(COLOR_RESET)"

# Compare Unigram vs BPE tokenizers - Intrinsic Quality Metrics
.PHONY: compare-tokenizers
compare-tokenizers: build-rust $(TOKENIZER_FILE) $(TOKENIZER_BPE_FILE)
	@echo ""
	$(CARGO) run --release --bin compare-tokenizers -- \
		--tokenizer1 $(TOKENIZER_FILE) \
		--tokenizer2 $(TOKENIZER_BPE_FILE) \
		--corpus $(CORPUS_DIR) \
		--name1 "Unigram" \
		--name2 "BPE" \
		--output tokenizer_comparison.md \
		--json tokenizer_comparison.json
	@echo ""
	@echo "$(COLOR_GREEN)Reports generated:$(COLOR_RESET)"
	@echo "  📄 Markdown: tokenizer_comparison.md"
	@echo "  📊 JSON:     tokenizer_comparison.json"
	@echo ""

# ============================================================================
# Text Generation
# ============================================================================

# Default values for generation
PROMPT      ?= Sherlock Holmes
MAX_TOKENS  ?= 100
SEED        ?=

generate: check-model
	@echo "$(COLOR_BOLD)Generating text...$(COLOR_RESET)"
	@echo "Prompt: $(PROMPT)"
	@echo "Max tokens: $(MAX_TOKENS)"
	@if [ -n "$(SEED)" ]; then \
		echo "Seed: $(SEED)"; \
		echo ""; \
		$(CARGO_RUN) generate \
			--prompt "$(PROMPT)" \
			--max-tokens $(MAX_TOKENS) \
			--seed $(SEED) \
			-m $(MODEL_FILE) \
			-t $(TOKENIZER_FILE); \
	else \
		echo ""; \
		$(CARGO_RUN) generate \
			--prompt "$(PROMPT)" \
			--max-tokens $(MAX_TOKENS) \
			-m $(MODEL_FILE) \
			-t $(TOKENIZER_FILE); \
	fi

# ============================================================================
# Query Trigram Probability
# ============================================================================

W1 ?= Sherlock
W2 ?= Holmes
W3 ?= was

query: check-model
	@echo "$(COLOR_BOLD)Querying trigram probability...$(COLOR_RESET)"
	$(CARGO_RUN) query \
		--w1 "$(W1)" \
		--w2 "$(W2)" \
		--w3 "$(W3)" \
		-m $(MODEL_FILE) \
		-t $(TOKENIZER_FILE)

# Serve web interface
# Usage: make serve [PORT=3000] [HOST=127.0.0.1]
PORT ?= 3000
HOST ?= 127.0.0.1

serve: check-model build-rust
	@echo "$(COLOR_BOLD)Starting web server on http://$(HOST):$(PORT)...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Visit the URL above in your browser to use the web interface$(COLOR_RESET)"
	@echo ""
	$(CARGO_WEB) --model $(MODEL_FILE) --tokenizer $(TOKENIZER_FILE) --port $(PORT) --host $(HOST)

# ============================================================================
# Build & Test
# ============================================================================

build-rust:
	@echo "$(COLOR_BOLD)Building Rust binaries...$(COLOR_RESET)"
	$(CARGO_BUILD)
	@echo "$(COLOR_GREEN)✓ Build complete$(COLOR_RESET)"

test:
	@echo "$(COLOR_BOLD)Running tests...$(COLOR_RESET)"
	$(CARGO) test
	@echo "$(COLOR_GREEN)✓ Tests complete$(COLOR_RESET)"

format:
	@echo "$(COLOR_BOLD)Formatting Rust code...$(COLOR_RESET)"
	$(CARGO) fmt
	@echo "$(COLOR_GREEN)✓ Format complete$(COLOR_RESET)"

lint:
	@echo "$(COLOR_BOLD)Running clippy linter...$(COLOR_RESET)"
	$(CARGO) clippy -- -D warnings
	@echo "$(COLOR_GREEN)✓ Lint complete$(COLOR_RESET)"

# ============================================================================
# Utilities
# ============================================================================

# Check if model exists
check-model:
	@if [ ! -f "$(MODEL_FILE)" ]; then \
		echo "$(COLOR_YELLOW)Warning: Model file not found: $(MODEL_FILE)$(COLOR_RESET)"; \
		echo "Run 'make build-model' first"; \
		exit 1; \
	fi

# Show configuration info
info:
	@echo "$(COLOR_BOLD)Configuration:$(COLOR_RESET)"
	@echo "  Corpus directory:  $(CORPUS_DIR)"
	@echo "  Data directory:    $(DATA_DIR)"
	@echo "  Model file:        $(MODEL_FILE)"
	@echo "  Tokenizer file:    $(TOKENIZER_FILE)"
	@echo "  Max memory:        $(MAX_MEMORY) MB"
	@echo "  Vocabulary size:   $(VOCAB_SIZE)"
	@echo ""
	@echo "$(COLOR_BOLD)Corpus files found:$(COLOR_RESET)"
	@if [ -z "$(TXT_FILES)" ]; then \
		echo "  $(COLOR_YELLOW)(none)$(COLOR_RESET)"; \
	else \
		find $(CORPUS_DIR) -type f -name "*.txt" 2>/dev/null | while read -r file; do \
			echo "  - $$file"; \
		done; \
	fi

# Clean generated files
clean:
	@echo "$(COLOR_BOLD)Cleaning...$(COLOR_RESET)"
	rm -f $(MODEL_FILE)
	$(CARGO) clean
	@echo "$(COLOR_GREEN)✓ Clean complete$(COLOR_RESET)"

# Clean everything including tokenizer
clean-all: clean
	rm -f $(TOKENIZER_FILE)
	@echo "$(COLOR_GREEN)✓ Deep clean complete$(COLOR_RESET)"

# ============================================================================
# Advanced Targets
# ============================================================================

# Quick test generation with default Sherlock Holmes prompt
demo: check-model
	@echo "$(COLOR_BOLD)Demo: Generating Sherlock Holmes text...$(COLOR_RESET)"
	@echo ""
	$(CARGO_RUN) generate \
		--prompt "Sherlock Holmes walked into" \
		--max-tokens 50 \
		--seed 42 \
		-m $(MODEL_FILE) \
		-t $(TOKENIZER_FILE)

# Build everything from scratch
rebuild: clean-all train-tokenizer build-model
	@echo "$(COLOR_GREEN)✓ Full rebuild complete$(COLOR_RESET)"
