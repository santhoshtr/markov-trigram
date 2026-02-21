mod direct_map;
mod hub_tokenizer;
mod sparse_trigram;
mod trigram_builder;
mod trigram_iterator;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use hub_tokenizer::{load_tokenizer, TokenizerType};
use markov_trigram::find_text_files;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use sparse_trigram::SparseTrigram;
use trigram_builder::TrigramBuilder;
#[derive(Parser)]
#[command(name = "markov-trigram")]
#[command(about = "A trigram language model implementation")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a trigram model from a corpus directory
    Build {
        /// Directory path to recursively search for text files
        #[arg(short = 'd', long = "dir", required = true)]
        directory: String,
        /// Output model file path
        #[arg(short, long, default_value = "trigram_model.bin")]
        output: String,
        /// Tokenizer type to download from HuggingFace Hub (ignored if -t is set)
        #[arg(long, default_value = "bpe")]
        tokenizer_type: TokenizerType,
        /// Path to a local tokenizer file (overrides --tokenizer-type)
        #[arg(short, long)]
        tokenizer: Option<String>,
        /// Maximum memory usage in MB
        #[arg(short, long, default_value = "1024")]
        max_memory: usize,
    },
    /// Query a trained model
    Query {
        /// Path to the model file
        #[arg(short, long, default_value = "trigram_model.bin")]
        model: String,
        /// Tokenizer type to download from HuggingFace Hub (ignored if -t is set)
        #[arg(long, default_value = "bpe")]
        tokenizer_type: TokenizerType,
        /// Path to a local tokenizer file (overrides --tokenizer-type)
        #[arg(short, long)]
        tokenizer: Option<String>,

        /// First word (w1)
        #[arg(long)]
        w1: String,
        /// Second word (w2)
        #[arg(long)]
        w2: String,
        /// Third word (w3)
        #[arg(long)]
        w3: String,
        /// Smoothing parameter
        #[arg(short, long)]
        smoothing: Option<f32>,
    },
    /// Generate text from a prompt using the trigram model
    Generate {
        /// Initial prompt text to start generation
        #[arg(short, long)]
        prompt: String,
        /// Path to the model file
        #[arg(short, long, default_value = "trigram_model.bin")]
        model: String,
        /// Tokenizer type to download from HuggingFace Hub (ignored if -t is set)
        #[arg(long, default_value = "bpe")]
        tokenizer_type: TokenizerType,
        /// Path to a local tokenizer file (overrides --tokenizer-type)
        #[arg(short, long)]
        tokenizer: Option<String>,
        /// Maximum number of tokens to generate
        #[arg(long, default_value = "1000")]
        max_tokens: usize,
        /// Random seed for reproducible generation
        #[arg(long)]
        seed: Option<u64>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Build {
            directory,
            output,
            tokenizer_type,
            tokenizer,
            max_memory,
        } => {
            let tok = load_tokenizer(tokenizer.as_deref(), &tokenizer_type)?;
            println!(
                "Building trigram model from corpus directory: {}",
                directory
            );
            println!();

            // Discover all text files in the directory
            let corpus_files = find_text_files(&directory)
                .map_err(|e| anyhow!("Failed to discover corpus files: {}", e))?;

            if corpus_files.is_empty() {
                return Err(anyhow!("No text files found in directory: {}", directory));
            }

            println!("Found {} text file(s) to process", corpus_files.len());
            println!();

            let mut builder = TrigramBuilder::new(tok, max_memory);

            // Process each corpus file with progress indicator
            let mut successful = 0;
            let mut failed = 0;
            let mut failed_files = Vec::new();

            for (idx, corpus_file) in corpus_files.iter().enumerate() {
                println!(
                    "[{}/{}] Processing: {}",
                    idx + 1,
                    corpus_files.len(),
                    corpus_file
                );

                match builder.process_corpus(corpus_file) {
                    Ok(_) => {
                        println!("  ✓ Completed: {}", corpus_file);
                        successful += 1;
                    }
                    Err(e) => {
                        eprintln!("  ⚠ Warning - Failed to process {}: {}", corpus_file, e);
                        eprintln!("  Continuing with remaining files...");
                        failed += 1;
                        failed_files.push(corpus_file.clone());
                    }
                }
                println!();
            }

            // Summary of processing
            println!("Processing complete:");
            println!("  ✓ Successfully processed: {} file(s)", successful);
            if failed > 0 {
                println!("  ⚠ Failed to process: {} file(s)", failed);
                for failed_file in &failed_files {
                    println!("    - {}", failed_file);
                }
            }
            println!();

            if successful == 0 {
                return Err(anyhow::anyhow!(
                    "No corpus files were successfully processed. Cannot build model."
                ));
            }

            println!(
                "Building final model from {} successful file(s)...",
                successful
            );
            let model = builder.build();

            println!("Model built with {} trigrams", model.total_trigrams);
            println!(
                "Memory usage: {:.2} MB",
                model.memory_usage() as f64 / 1024.0 / 1024.0
            );

            model.save(&output)?;
            println!("Model saved to: {}", output);
        }
        Commands::Query {
            model,
            tokenizer_type,
            tokenizer,
            w1,
            w2,
            w3,
            smoothing,
        } => {
            println!("Loading model from: {}", model);
            let loaded_model = SparseTrigram::load(&model)?;
            let tok = load_tokenizer(tokenizer.as_deref(), &tokenizer_type)?;
            println!("Querying P({} | {}, {})", w3, w1, w2);
            let tokens = tok.encode(format!("{} {} {}", w1, w2, w3), false).unwrap();
            // Take the last 3 tokens as w1, w2, w3
            let token_ids = tokens.get_ids();
            if token_ids.len() < 3 {
                println!("Input words not found in tokenizer vocabulary.");
                return Ok(());
            }
            let w1_id = token_ids[token_ids.len() - 3];
            let w2_id = token_ids[token_ids.len() - 2];
            let w3_id = token_ids[token_ids.len() - 1];
            let prob = loaded_model.probability(w1_id, w2_id, w3_id, smoothing);
            println!("P({} | {}, {}) = {}", w3, w1, w2, prob);

            let count = loaded_model.get_count(w1_id, w2_id, w3_id);
            println!("Count: {}", count);
        }
        Commands::Generate {
            model,
            tokenizer_type,
            tokenizer,
            prompt,
            max_tokens,
            seed,
        } => {
            // Load model and tokenizer
            let loaded_model = SparseTrigram::load(&model)?;
            let tok = load_tokenizer(tokenizer.as_deref(), &tokenizer_type)?;

            // Validate prompt
            if prompt.trim().is_empty() {
                return Err(anyhow!("Prompt cannot be empty"));
            }

            // Tokenize the prompt
            let encoding = tok
                .encode(prompt.as_str(), false)
                .map_err(|e| anyhow!("Failed to tokenize prompt: {}", e))?;
            let token_ids = encoding.get_ids();

            // Validate we have at least 2 tokens
            if token_ids.len() < 2 {
                return Err(anyhow!(
                    "Prompt must tokenize to at least 2 tokens. Got {} token(s).",
                    token_ids.len()
                ));
            }

            // Extract last 2 tokens as initial context
            let mut w1 = token_ids[token_ids.len() - 2];
            let mut w2 = token_ids[token_ids.len() - 1];

            // Store all tokens (prompt + generated)
            let mut all_tokens: Vec<u32> = token_ids.to_vec();

            // Generation loop with seed-based or random RNG
            match seed {
                Some(s) => {
                    let mut rng = StdRng::seed_from_u64(s);
                    for _ in 0..max_tokens {
                        let next_token = match loaded_model.sample_next(w1, w2, &mut rng) {
                            Some(token) => token,
                            None => rng.random_range(0..loaded_model.vocabulary_size as u32),
                        };
                        all_tokens.push(next_token);
                        w1 = w2;
                        w2 = next_token;
                    }
                }
                None => {
                    let mut rng = rand::rng();
                    for _ in 0..max_tokens {
                        let next_token = match loaded_model.sample_next(w1, w2, &mut rng) {
                            Some(token) => token,
                            None => rng.random_range(0..loaded_model.vocabulary_size as u32),
                        };
                        all_tokens.push(next_token);
                        w1 = w2;
                        w2 = next_token;
                    }
                }
            }

            // Decode all tokens to text
            let generated_text = tok
                .decode(&all_tokens, true)
                .map_err(|e| anyhow!("Failed to decode tokens: {}", e))?;

            // Output the generated text
            println!("{}", generated_text);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use tokenizers::Tokenizer;

    // Helper function to create a minimal test tokenizer
    fn create_test_tokenizer() -> Result<(Tokenizer, NamedTempFile)> {
        let mut temp_file = NamedTempFile::new()?;

        // Minimal tokenizer JSON that should work with tokenizers crate
        let tokenizer_json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 3, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "Unigram",
                "unk_id": 3,
                "vocab": [
                    ["<s>", 0.0],
                    ["<pad>", 0.0],
                    ["</s>", 0.0],
                    ["<unk>", 0.0],
                    ["test", -1.0],
                    ["word", -1.0],
                    ["1", -1.0],
                    ["2", -1.0],
                    ["3", -1.0],
                    ["4", -1.0],
                    ["5", -1.0]
                ]
            }
        }"#;

        temp_file.write_all(tokenizer_json.as_bytes())?;
        temp_file.flush()?;
        let path = temp_file.path().to_str().unwrap().to_string();
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| anyhow!("Failed to load test tokenizer: {e}"))?;
        Ok((tokenizer, temp_file))
    }

    #[test]
    fn test_trigram_storage() -> Result<()> {
        let (tokenizer, _file) = create_test_tokenizer()?;
        let mut builder = TrigramBuilder::new(tokenizer, 10);

        // Add some trigrams directly
        builder.add_trigram(1, 2, 3);
        builder.add_trigram(1, 2, 3); // Duplicate
        builder.add_trigram(1, 2, 4);
        builder.add_trigram(1, 3, 5);
        builder.add_trigram(2, 3, 4);

        let model = builder.build();

        assert_eq!(model.get_count(1, 2, 3), 2);
        assert_eq!(model.get_count(1, 2, 4), 1);
        assert_eq!(model.get_count(1, 3, 5), 1);
        assert_eq!(model.get_count(2, 3, 4), 1);
        assert_eq!(model.get_count(99, 99, 99), 0); // Non-existent

        Ok(())
    }

    #[test]
    fn test_probability_calculation() -> Result<()> {
        let (tokenizer, _file) = create_test_tokenizer()?;
        let mut builder = TrigramBuilder::new(tokenizer, 10);

        // Add trigrams to create known probabilities
        for _ in 0..3 {
            builder.add_trigram(1, 2, 3);
        }
        builder.add_trigram(1, 2, 4);

        let model = builder.build();

        // P(3 | 1, 2) = 3/4 = 0.75
        let prob = model.probability(1, 2, 3, None);
        assert!((prob - 0.75).abs() < 0.001);

        // P(4 | 1, 2) = 1/4 = 0.25
        let prob = model.probability(1, 2, 4, None);
        assert!((prob - 0.25).abs() < 0.001);

        // Non-existent trigram should have probability 0 without smoothing
        let prob = model.probability(1, 2, 99, None);
        assert!(prob.abs() < 0.001);

        // With smoothing, should get non-zero probability
        let prob = model.probability(1, 2, 99, Some(0.1));
        assert!(prob > 0.0);

        Ok(())
    }

    #[test]
    fn test_sampling() -> Result<()> {
        let (tokenizer, _file) = create_test_tokenizer()?;
        let mut builder = TrigramBuilder::new(tokenizer, 10);

        builder.add_trigram(1, 2, 3);
        builder.add_trigram(1, 2, 4);
        builder.add_trigram(1, 2, 4); // Make 4 more likely

        let model = builder.build();
        let mut rng = StdRng::seed_from_u64(42);

        // Count samples
        let mut counts = [0; 5];
        for _ in 0..1000 {
            if let Some(w3) = model.sample_next(1, 2, &mut rng) {
                if w3 == 3 {
                    counts[3_usize] += 1;
                } else if w3 == 4 {
                    counts[4_usize] += 1;
                }
            }
        }

        // 4 should be sampled about twice as often as 3
        let ratio = counts[4] as f32 / counts[3] as f32;
        assert!(ratio > 1.5 && ratio < 2.5); // Allow some variance

        Ok(())
    }
}
