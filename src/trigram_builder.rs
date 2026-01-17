use anyhow::Result;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::sparse_trigram::SparseTrigram;

// Builder for incremental construction
pub struct TrigramBuilder {
    // Temporary storage: HashMap<w1, BTreeMap<w2, HashMap<w3, count>>>
    data: HashMap<u32, BTreeMap<u32, HashMap<u32, u32>>>,
    tokenizer: Tokenizer,
    vocabulary_size: u32,
    max_memory_bytes: usize,
}

impl TrigramBuilder {
    pub fn new(tokenizer_path: &str, max_memory_mb: usize) -> Self {
        let tokenizer: Tokenizer = Tokenizer::from_file(tokenizer_path).unwrap();
        let vocabulary_size = tokenizer.get_vocab_size(true) as u32;

        Self {
            data: HashMap::new(),
            tokenizer,
            vocabulary_size,
            max_memory_bytes: max_memory_mb * 1024 * 1024,
        }
    }

    /// Add a trigram to the builder (used for manual construction and tests)
    #[allow(dead_code)]
    pub fn add_trigram(&mut self, w1: u32, w2: u32, w3: u32) {
        let w1_entry = self.data.entry(w1).or_default();
        let w2_entry = w1_entry.entry(w2).or_default();
        *w2_entry.entry(w3).or_insert(0) += 1;

        // Check memory and flush if needed
        if self.estimate_memory() > self.max_memory_bytes {
            // In production, you'd flush to disk here
            // For now, we'll just warn
            eprintln!("Warning: Approaching memory limit");
        }
    }

    /// Process a corpus file with parallel processing
    ///
    /// Reads the corpus in chunks and processes each chunk in parallel using all available CPU cores.
    /// Uses Rayon for parallel iteration over lines within each chunk.
    ///
    /// # Arguments
    /// * `corpus_path` - Path to the text corpus file
    ///
    /// # Performance
    /// - Processes 10,000 lines per chunk to balance memory usage and parallelism
    /// - Each line is tokenized and processed independently in parallel
    /// - Results are merged progressively to respect memory limits
    pub fn process_corpus<P: AsRef<Path>>(&mut self, corpus_path: P) -> Result<()> {
        const CHUNK_SIZE: usize = 10000; // Process 10k lines at a time

        let file = File::open(corpus_path)?;
        let reader = BufReader::new(file);

        let mut lines_buffer = Vec::with_capacity(CHUNK_SIZE);
        let tokenizer = Arc::new(self.tokenizer.clone());

        for line in reader.lines() {
            let line = line?;
            lines_buffer.push(line);

            // Process chunk when buffer is full
            if lines_buffer.len() >= CHUNK_SIZE {
                let local_trigrams = self.process_lines_parallel(&lines_buffer, &tokenizer);
                self.merge_trigrams(local_trigrams);
                lines_buffer.clear();
            }
        }

        // Process remaining lines
        if !lines_buffer.is_empty() {
            let local_trigrams = self.process_lines_parallel(&lines_buffer, &tokenizer);
            self.merge_trigrams(local_trigrams);
        }

        Ok(())
    }

    /// Process a batch of lines in parallel, returning local trigram counts
    fn process_lines_parallel(
        &self,
        lines: &[String],
        tokenizer: &Arc<Tokenizer>,
    ) -> Vec<HashMap<(u32, u32, u32), u32>> {
        lines
            .par_iter()
            .map(|line| {
                let mut local_map = HashMap::new();

                // Tokenize the entire line at once (more efficient)
                if let Ok(encoded) = tokenizer.encode(line.as_str(), false) {
                    let word_ids = encoded.get_ids();

                    // Extract trigrams from this line
                    for i in 2..word_ids.len() {
                        let trigram = (word_ids[i - 2], word_ids[i - 1], word_ids[i]);
                        *local_map.entry(trigram).or_insert(0) += 1;
                    }
                }

                local_map
            })
            .collect()
    }

    /// Merge local trigram counts into the main data structure
    fn merge_trigrams(&mut self, local_trigrams: Vec<HashMap<(u32, u32, u32), u32>>) {
        for local_map in local_trigrams {
            for ((w1, w2, w3), count) in local_map {
                let w1_entry = self.data.entry(w1).or_default();
                let w2_entry = w1_entry.entry(w2).or_default();
                *w2_entry.entry(w3).or_insert(0) += count;
            }
        }

        // Check memory after merge
        if self.estimate_memory() > self.max_memory_bytes {
            eprintln!("Warning: Approaching memory limit");
        }
    }

    /// Build the final compressed sparse fiber representation
    pub fn build(self) -> SparseTrigram {
        let mut model = SparseTrigram::new(self.vocabulary_size as usize);

        // Sort w1 keys for deterministic ordering
        let mut w1_keys: Vec<u32> = self.data.keys().copied().collect();
        w1_keys.sort_unstable();

        // Create mapping from w1 values to their indices
        let mut w1_to_idx = HashMap::new();
        for (idx, &w1) in w1_keys.iter().enumerate() {
            w1_to_idx.insert(w1, idx);
        }

        // Build CSF structure
        let mut w1_ptr = vec![0];
        let mut w2_indices = Vec::new();
        let mut w2_ptr = Vec::new();
        let mut w3_indices = Vec::new();
        let mut counts = Vec::new();

        for &w1 in &w1_keys {
            let w2_map = &self.data[&w1];

            // Sort w2 keys for this w1
            let mut w2_keys: Vec<u32> = w2_map.keys().copied().collect();
            w2_keys.sort_unstable();

            for &w2 in &w2_keys {
                let w3_map = &w2_map[&w2];
                w2_indices.push(w2);

                // Record start of w3 entries for this (w1, w2)
                w2_ptr.push(w3_indices.len());

                // Sort w3 keys for this (w1, w2)
                let mut w3_entries: Vec<(u32, u32)> =
                    w3_map.iter().map(|(&w3, &count)| (w3, count)).collect();
                w3_entries.sort_unstable_by_key(|&(w3, _)| w3);

                // Add w3 indices and counts
                for (w3, count) in w3_entries {
                    w3_indices.push(w3);
                    counts.push(count);

                    // Update totals for smoothing
                    let bigram_idx = w1 as usize * model.vocabulary_size + w2 as usize;
                    model.bigram_totals[bigram_idx] += count;
                    model.unigram_totals[w1 as usize] += count;
                    model.total_trigrams += 1;
                }
            }

            // Record end pointer for this w1
            w1_ptr.push(w2_indices.len());
        }

        // Add final pointer for w2_ptr
        w2_ptr.push(w3_indices.len());

        // Update model with built data
        model.w1_to_idx = w1_to_idx;
        model.w1_ptr = w1_ptr;
        model.w2_indices = w2_indices;
        model.w2_ptr = w2_ptr;
        model.w3_indices = w3_indices;
        model.counts = counts;

        model
    }

    /// Estimate current memory usage
    fn estimate_memory(&self) -> usize {
        let mut total = 0;

        for w2_map in self.data.values() {
            total += std::mem::size_of::<u32>(); // w1
            total += w2_map.len() * std::mem::size_of::<u32>(); // w2 keys

            for w3_map in w2_map.values() {
                total += std::mem::size_of::<u32>(); // w2
                total += w3_map.len() * (std::mem::size_of::<u32>() * 2); // w3 keys + counts
            }
        }

        total
    }
}
