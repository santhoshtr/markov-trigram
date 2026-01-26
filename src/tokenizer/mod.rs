// Tokenizer implementations for training and encoding text
//
// This module contains different tokenizer variants for processing text corpora:
// - Unigram tokenizer: Statistical tokenizer based on unigram model
// - BPE tokenizer: Byte Pair Encoding tokenizer
// - Evaluation: Intrinsic quality metrics for comparing tokenizers

pub mod evaluation;

// Re-export find_text_files for use by tokenizer binaries
pub use crate::find_text_files;
