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
CARGO_TOK      := $(CARGO) run --release --bin ml_tokenizer --

# Find all .txt files recursively in CORPUS_DIR
TXT_FILES      := $(shell find $(CORPUS_DIR) -type f -name "*.txt" 2>/dev/null)

# Generate corpus arguments for multiple files
CORPUS_ARGS    := $(foreach file,$(TXT_FILES),-c $(file))

# Colors for output
COLOR_RESET    := \033[0m
COLOR_BOLD     := \033[1m
COLOR_GREEN    := \033[32m
COLOR_YELLOW   := \033[33m
COLOR_BLUE     := \033[34m

# ============================================================================
# Main Targets
# ============================================================================

.PHONY: all help build test clean

# Default target
all: help

# Show help message
help:
	@echo ""
	@echo "$(COLOR_BOLD)Markov Trigram Project - Makefile$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Available targets:$(COLOR_RESET)"
	@echo "  $(COLOR_GREEN)help$(COLOR_RESET)              - Show this help message"
	@echo "  $(COLOR_GREEN)build-model$(COLOR_RESET)       - Build trigram model from all corpus files"
	@echo "  $(COLOR_GREEN)train-tokenizer$(COLOR_RESET)   - Train tokenizer from corpus directory"
	@echo "  $(COLOR_GREEN)generate$(COLOR_RESET)          - Generate text from a prompt (use PROMPT variable)"
	@echo "  $(COLOR_GREEN)query$(COLOR_RESET)             - Query trigram probability (use W1, W2, W3 variables)"
	@echo "  $(COLOR_GREEN)test$(COLOR_RESET)              - Run Rust tests"
	@echo "  $(COLOR_GREEN)build$(COLOR_RESET)             - Build Rust binaries"
	@echo "  $(COLOR_GREEN)format$(COLOR_RESET)            - Format Rust code with cargo fmt"
	@echo "  $(COLOR_GREEN)lint$(COLOR_RESET)              - Run clippy linter"
	@echo "  $(COLOR_GREEN)clean$(COLOR_RESET)             - Clean build artifacts and generated files"
	@echo "  $(COLOR_GREEN)clean-all$(COLOR_RESET)         - Deep clean including tokenizer"
	@echo "  $(COLOR_GREEN)info$(COLOR_RESET)              - Show configuration info"
	@echo "  $(COLOR_GREEN)demo$(COLOR_RESET)              - Quick demo with Sherlock Holmes prompt"
	@echo ""
	@echo "$(COLOR_BOLD)Examples:$(COLOR_RESET)"
	@echo "  make build-model"
	@echo "  make generate PROMPT=\"Sherlock Holmes\" MAX_TOKENS=100"
	@echo "  make generate PROMPT=\"Watson said\" SEED=42"
	@echo "  make query W1=Sherlock W2=Holmes W3=was"
	@echo "  make train-tokenizer VOCAB_SIZE=20000"
	@echo ""

# ============================================================================
# Model Building
# ============================================================================

build-model: build-rust
	@echo "$(COLOR_BOLD)Building trigram model...$(COLOR_RESET)"
	@echo "Corpus directory: $(CORPUS_DIR)"
	@NUM_FILES=$$(find $(CORPUS_DIR) -type f -name "*.txt" 2>/dev/null | wc -l); \
	echo "Found $$NUM_FILES file(s)"; \
	if [ $$NUM_FILES -eq 0 ]; then \
		echo "$(COLOR_YELLOW)Warning: No .txt files found in $(CORPUS_DIR)$(COLOR_RESET)"; \
		exit 1; \
	fi; \
	echo ""; \
	CORPUS_FILES=$$(find $(CORPUS_DIR) -type f -name "*.txt" 2>/dev/null | while IFS= read -r file; do printf ' -c '\''%s'\''' "$$file"; done); \
	eval $(CARGO_RUN) build $$CORPUS_FILES -o $(MODEL_FILE) -t $(TOKENIZER_FILE) -m $(MAX_MEMORY)
	@echo ""
	@echo "$(COLOR_GREEN)✓ Model built successfully: $(MODEL_FILE)$(COLOR_RESET)"

# ============================================================================
# Tokenizer Training
# ============================================================================

train-tokenizer:
	@echo "$(COLOR_BOLD)Training tokenizer...$(COLOR_RESET)"
	@echo "Corpus directory: $(CORPUS_DIR)"
	@echo "Vocabulary size: $(VOCAB_SIZE)"
	@echo "Output: $(TOKENIZER_FILE)"
	@mkdir -p $(DATA_DIR)
	$(CARGO_TOK) train \
		-f $(CORPUS_DIR) \
		-v $(VOCAB_SIZE) \
		-o $(TOKENIZER_FILE)
	@echo ""
	@echo "$(COLOR_GREEN)✓ Tokenizer trained: $(TOKENIZER_FILE)$(COLOR_RESET)"

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
