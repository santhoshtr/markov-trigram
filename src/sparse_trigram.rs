use crate::trigram_iterator::TrigramIterator;
use anyhow::{anyhow, Result};
use rayon::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

use std::fs::File;
use std::io::{Read, Write};

// Compressed Sparse Fiber representation for trigrams
#[derive(Archive, Deserialize, Serialize, Clone)]
pub struct SparseTrigram {
    // Mapping from w1 values to their indices in the sorted arrays
    pub w1_to_idx: HashMap<u32, usize>,

    // Level 1: w1 -> range of w2 entries
    pub w1_ptr: Vec<usize>,

    // Level 2: w2 indices within each w1
    pub w2_indices: Vec<u32>,

    // Level 2: w2 -> range of w3 entries
    pub w2_ptr: Vec<usize>,

    // Level 3: w3 indices and their frequencies
    pub w3_indices: Vec<u32>,
    pub counts: Vec<u32>,

    // Metadata
    pub vocabulary_size: usize,
    pub total_trigrams: usize,
    // For backoff smoothing
    pub bigram_totals: Vec<u32>,  // Total count for each (w1, w2) pair
    pub unigram_totals: Vec<u32>, // Total count for each w1
}

impl SparseTrigram {
    /// Create a new empty trigram model
    pub fn new(vocabulary_size: usize) -> Self {
        Self {
            w1_to_idx: HashMap::new(),
            w1_ptr: vec![0], // Single element, will expand
            w2_indices: Vec::new(),
            w2_ptr: Vec::new(),
            w3_indices: Vec::new(),
            counts: Vec::new(),
            vocabulary_size,
            total_trigrams: 0,
            bigram_totals: vec![0; vocabulary_size * vocabulary_size],
            unigram_totals: vec![0; vocabulary_size],
        }
    }

    /// Get frequency count for a specific trigram (w1, w2, w3)
    pub fn get_count(&self, w1: u32, w2: u32, w3: u32) -> u32 {
        let w1_idx = match self.w1_to_idx.get(&w1) {
            Some(&idx) => idx,
            None => return 0,
        };

        // Check bounds
        if w1_idx >= self.w1_ptr.len() - 1 {
            return 0;
        }

        // Get range of w2 indices for this w1
        let w2_start = self.w1_ptr[w1_idx];
        let w2_end = self.w1_ptr[w1_idx + 1];

        if w2_start == w2_end {
            return 0;
        }

        // Binary search for w2 within this w1's w2_indices
        let w2_slice = &self.w2_indices[w2_start..w2_end];
        match w2_slice.binary_search(&w2) {
            Ok(w2_rel_idx) => {
                let w2_abs_idx = w2_start + w2_rel_idx;

                // Get range of w3 indices for this (w1, w2) pair
                let w3_start = self.w2_ptr[w2_abs_idx];
                let w3_end = self.w2_ptr[w2_abs_idx + 1];

                if w3_start == w3_end {
                    return 0;
                }

                // Binary search for w3 within this (w1, w2)'s w3_indices
                let w3_slice = &self.w3_indices[w3_start..w3_end];
                match w3_slice.binary_search(&w3) {
                    Ok(w3_rel_idx) => self.counts[w3_start + w3_rel_idx],
                    Err(_) => 0,
                }
            }
            Err(_) => 0,
        }
    }

    /// Get all possible w3s for a given (w1, w2) context with their frequencies
    pub fn get_w3_candidates(&self, w1: u32, w2: u32) -> Vec<(u32, u32)> {
        let w1_idx = match self.w1_to_idx.get(&w1) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        if w1_idx >= self.w1_ptr.len() - 1 {
            return Vec::new();
        }

        let w2_start = self.w1_ptr[w1_idx];
        let w2_end = self.w1_ptr[w1_idx + 1];

        if w2_start == w2_end {
            return Vec::new();
        }

        let w2_slice = &self.w2_indices[w2_start..w2_end];
        match w2_slice.binary_search(&w2) {
            Ok(w2_rel_idx) => {
                let w2_abs_idx = w2_start + w2_rel_idx;
                let w3_start = self.w2_ptr[w2_abs_idx];
                let w3_end = self.w2_ptr[w2_abs_idx + 1];

                let mut results = Vec::with_capacity(w3_end - w3_start);
                for i in w3_start..w3_end {
                    results.push((self.w3_indices[i], self.counts[i]));
                }
                results
            }
            Err(_) => Vec::new(),
        }
    }

    /// Calculate probability P(w3 | w1, w2) with optional smoothing
    pub fn probability(&self, w1: u32, w2: u32, w3: u32, smoothing: Option<f32>) -> f32 {
        let count = self.get_count(w1, w2, w3) as f32;

        if count > 0.0 {
            // Get total count for this (w1, w2) context
            let bigram_idx = w1 as usize * self.vocabulary_size + w2 as usize;
            let total = self.bigram_totals[bigram_idx] as f32;

            if total > 0.0 {
                return count / total;
            }
        }

        // Apply smoothing if requested
        if let Some(alpha) = smoothing {
            let unigram_idx = w1 as usize;
            let unigram_total = self.unigram_totals[unigram_idx] as f32;
            let vocab_size = self.vocabulary_size as f32;

            if unigram_total > 0.0 {
                // Laplace smoothing: (count + alpha) / (total + alpha * vocab_size)
                alpha / (unigram_total + alpha * vocab_size)
            } else {
                1.0 / vocab_size // Uniform distribution
            }
        } else {
            0.0
        }
    }

    /// Sample next word given context (w1, w2)
    pub fn sample_next(&self, w1: u32, w2: u32, rng: &mut impl rand::RngExt) -> Option<u32> {
        let candidates = self.get_w3_candidates(w1, w2);

        if candidates.is_empty() {
            return None;
        }

        // Calculate total for this context
        let total: u32 = candidates.iter().map(|&(_, count)| count).sum();

        // Roulette wheel selection
        let mut cumsum = 0;
        let rand_val: u32 = rng.random_range(0..total);

        for (w3, count) in candidates {
            cumsum += count;
            if rand_val < cumsum {
                return Some(w3);
            }
        }

        // Fallback
        self.get_w3_candidates(w1, w2).first().map(|&(w3, _)| w3)
    }

    /// Save model to disk
    pub fn save(&self, path: &str) -> Result<()> {
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map_err(|e| anyhow!("Serialization failed: {}", e))?;

        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load model from disk
    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        // Use from_bytes which handles deserialization correctly
        let model: SparseTrigram = rkyv::from_bytes::<SparseTrigram, rkyv::rancor::Error>(&bytes)
            .map_err(|e| anyhow!("Deserialization failed: {}", e))?;

        Ok(model)
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.w1_to_idx.len() * (std::mem::size_of::<u32>() + std::mem::size_of::<usize>())
            + self.w1_ptr.len() * std::mem::size_of::<usize>()
            + self.w2_indices.len() * std::mem::size_of::<u32>()
            + self.w2_ptr.len() * std::mem::size_of::<usize>()
            + self.w3_indices.len() * std::mem::size_of::<u32>()
            + self.counts.len() * std::mem::size_of::<u32>()
            + self.bigram_totals.len() * std::mem::size_of::<u32>()
            + self.unigram_totals.len() * std::mem::size_of::<u32>()
    }

    /// Get iterator over all trigrams
    pub fn iter(&self) -> TrigramIterator<'_> {
        TrigramIterator {
            model: self,
            w1_idx: 0,
            w2_rel_idx: 0,
            w3_idx: 0,
        }
    }

    /// Parallel processing of queries (useful for batch scoring)
    pub fn batch_probabilities(&self, queries: &[(u32, u32, u32)], smoothing: f32) -> Vec<f32> {
        queries
            .par_iter()
            .map(|&(w1, w2, w3)| self.probability(w1, w2, w3, Some(smoothing)))
            .collect()
    }
}
