use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Results from evaluating a tokenizer on intrinsic quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    // Tier 1: Most Predictive
    /// Rényi entropy (α=2) - measures token distribution diversity
    pub renyi_entropy: f64,
    /// Fertility - average tokens per word
    pub fertility: f64,

    // Tier 2: Standard Efficiency
    /// Compression rate - characters per token
    pub compression_rate: f64,
    /// Vocabulary utilization - % of vocab actually used
    pub vocab_utilization: f64,
    /// OOV rate - % of unknown tokens
    pub oov_rate: f64,
    /// Encoding speed - characters per second
    pub encoding_speed_chars_per_sec: f64,

    // Tier 3: Malayalam-Specific
    /// Morphological consistency - % of word families with shared root token
    pub morphological_consistency: f64,
    /// Rare word handling ratio - (avg tokens for rare) / (avg tokens for common)
    pub rare_word_handling_ratio: f64,
    /// Shannon entropy - baseline entropy metric
    pub shannon_entropy: f64,

    // Metadata
    pub total_tokens: usize,
    pub unique_tokens: usize,
    pub total_characters: usize,
    pub total_words: usize,
    pub vocab_size: usize,
}

/// Comparison results between two tokenizers
#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonResults {
    pub tokenizer1_name: String,
    pub tokenizer2_name: String,
    pub results1: EvaluationResults,
    pub results2: EvaluationResults,
    pub winner: String,
    pub metrics_won: (usize, usize), // (tokenizer1 wins, tokenizer2 wins)
}

/// Token statistics collected during evaluation
#[derive(Debug)]
struct TokenStats {
    token_frequencies: HashMap<u32, usize>,
    total_tokens: usize,
    unique_tokens: usize,
    unk_count: usize,
}

/// Evaluate a tokenizer on all 9 intrinsic quality metrics
pub fn evaluate_tokenizer(tokenizer: &Tokenizer, corpus: &str) -> Result<EvaluationResults> {
    eprintln!("  Tokenizing corpus...");
    let start_encode = Instant::now();
    let encoded = tokenizer
        .encode(corpus, false)
        .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;
    let tokens = encoded.get_ids().to_vec();
    let encoding_time = start_encode.elapsed();

    eprintln!("  Collecting token statistics...");
    let token_stats = collect_token_stats(&tokens);

    let vocab_size = tokenizer.get_vocab_size(true);
    let _unk_token_id = tokenizer.token_to_id("<unk>").unwrap_or(3);

    eprintln!("  Calculating metrics...");

    let total_chars = corpus.chars().count();
    let total_words = count_words(corpus);

    Ok(EvaluationResults {
        // Tier 1
        renyi_entropy: calculate_renyi_entropy(
            &token_stats.token_frequencies,
            token_stats.total_tokens,
        ),
        fertility: if total_words > 0 {
            tokens.len() as f64 / total_words as f64
        } else {
            0.0
        },

        // Tier 2
        compression_rate: if !tokens.is_empty() {
            total_chars as f64 / tokens.len() as f64
        } else {
            0.0
        },
        vocab_utilization: (token_stats.unique_tokens as f64 / vocab_size as f64) * 100.0,
        oov_rate: (token_stats.unk_count as f64 / token_stats.total_tokens as f64) * 100.0,
        encoding_speed_chars_per_sec: total_chars as f64 / encoding_time.as_secs_f64(),

        // Tier 3
        morphological_consistency: test_morphological_consistency(tokenizer)?,
        rare_word_handling_ratio: test_rare_word_handling(tokenizer, corpus)?,
        shannon_entropy: calculate_shannon_entropy(
            &token_stats.token_frequencies,
            token_stats.total_tokens,
        ),

        // Metadata
        total_tokens: token_stats.total_tokens,
        unique_tokens: token_stats.unique_tokens,
        total_characters: total_chars,
        total_words,
        vocab_size,
    })
}

/// Calculate Rényi entropy (α=2)
/// Formula: H₂(X) = -log₂(Σᵢ p(xᵢ)²)
/// Higher is better
fn calculate_renyi_entropy(token_frequencies: &HashMap<u32, usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }

    let mut sum_squared_probs = 0.0;

    for &count in token_frequencies.values() {
        let prob = count as f64 / total as f64;
        sum_squared_probs += prob * prob;
    }

    if sum_squared_probs > 0.0 {
        -sum_squared_probs.log2()
    } else {
        0.0
    }
}

/// Calculate Shannon entropy
/// Formula: H(X) = -Σᵢ p(xᵢ) log₂ p(xᵢ)
/// Higher is better
fn calculate_shannon_entropy(token_frequencies: &HashMap<u32, usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }

    let mut entropy = 0.0;

    for &count in token_frequencies.values() {
        let prob = count as f64 / total as f64;
        if prob > 0.0 {
            entropy -= prob * prob.log2();
        }
    }

    entropy
}

/// Test morphological consistency for Malayalam word families
/// Returns % of families where root token is shared
fn test_morphological_consistency(tokenizer: &Tokenizer) -> Result<f64> {
    let word_families = get_malayalam_word_families();

    let mut consistent_families = 0;
    let total_families = word_families.len();

    for family in &word_families {
        if has_consistent_root(tokenizer, family)? {
            consistent_families += 1;
        }
    }

    Ok((consistent_families as f64 / total_families as f64) * 100.0)
}

/// Check if a word family has a consistent first token (root)
fn has_consistent_root(tokenizer: &Tokenizer, words: &[&str]) -> Result<bool> {
    let mut root_tokens = Vec::new();

    for word in words {
        let encoded = tokenizer
            .encode(*word, false)
            .map_err(|e| anyhow::anyhow!("Encoding failed for '{}': {}", word, e))?;
        let tokens = encoded.get_ids();
        if !tokens.is_empty() {
            root_tokens.push(tokens[0]); // First token is often the root
        }
    }

    // Check if all words share the same first token
    if root_tokens.is_empty() {
        return Ok(false);
    }

    let first = root_tokens[0];
    Ok(root_tokens.iter().all(|&t| t == first))
}

/// Get Malayalam word families for morphological testing
fn get_malayalam_word_families() -> Vec<Vec<&'static str>> {
    vec![
        vec!["കേരളം", "കേരളത്തിൽ", "കേരളത്തിന്", "കേരളത്തിലെ"],
        vec!["മലയാളം", "മലയാളത്തിൽ", "മലയാളിക്ക്"],
        vec!["വീട്", "വീട്ടിൽ", "വീട്ടിലേക്ക്", "വീട്ടിൽനിന്ന്"],
        vec!["പുസ്തകം", "പുസ്തകത്തിൽ", "പുസ്തകങ്ങൾ"],
        vec!["കുട്ടി", "കുട്ടികൾ", "കുട്ടിയുടെ"],
    ]
}

/// Test rare word handling
/// Returns ratio: (avg tokens for rare) / (avg tokens for common)
/// Closer to 1.0 is better (consistent regardless of frequency)
fn test_rare_word_handling(tokenizer: &Tokenizer, corpus: &str) -> Result<f64> {
    let word_frequencies = count_word_frequencies(corpus);

    if word_frequencies.len() < 20 {
        // Not enough words to test
        return Ok(1.0);
    }

    // Identify rare words (frequency < 5)
    let rare_words: Vec<_> = word_frequencies
        .iter()
        .filter(|(_, &count)| count < 5)
        .map(|(word, _)| word.as_str())
        .take(100)
        .collect();

    // Identify common words (frequency >= 100)
    let common_words: Vec<_> = word_frequencies
        .iter()
        .filter(|(_, &count)| count >= 100)
        .map(|(word, _)| word.as_str())
        .take(100)
        .collect();

    if rare_words.is_empty() || common_words.is_empty() {
        return Ok(1.0);
    }

    let avg_rare = average_tokens_per_word(tokenizer, &rare_words)?;
    let avg_common = average_tokens_per_word(tokenizer, &common_words)?;

    if avg_common > 0.0 {
        Ok(avg_rare / avg_common)
    } else {
        Ok(1.0)
    }
}

/// Count word frequencies in corpus
fn count_word_frequencies(corpus: &str) -> HashMap<String, usize> {
    let mut frequencies = HashMap::new();
    for word in corpus.split_whitespace() {
        if !word.is_empty() {
            *frequencies.entry(word.to_string()).or_insert(0) += 1;
        }
    }
    frequencies
}

/// Calculate average tokens per word for a list of words
fn average_tokens_per_word(tokenizer: &Tokenizer, words: &[&str]) -> Result<f64> {
    if words.is_empty() {
        return Ok(0.0);
    }

    let mut total_tokens = 0;
    for word in words {
        let encoded = tokenizer
            .encode(*word, false)
            .map_err(|e| anyhow::anyhow!("Encoding failed for '{}': {}", word, e))?;
        total_tokens += encoded.get_ids().len();
    }
    Ok(total_tokens as f64 / words.len() as f64)
}

/// Count number of words (whitespace-separated)
fn count_words(text: &str) -> usize {
    text.split_whitespace().filter(|s| !s.is_empty()).count()
}

/// Collect token statistics from token sequence
fn collect_token_stats(tokens: &[u32]) -> TokenStats {
    let mut token_frequencies = HashMap::new();
    let mut unk_count = 0;

    for &token in tokens {
        *token_frequencies.entry(token).or_insert(0) += 1;
        // Token ID 3 is typically <unk> (follows HuggingFace convention)
        if token == 3 {
            unk_count += 1;
        }
    }

    TokenStats {
        unique_tokens: token_frequencies.len(),
        total_tokens: tokens.len(),
        unk_count,
        token_frequencies,
    }
}

/// Determine which tokenizer wins overall
/// Returns positive if tokenizer1 wins, negative if tokenizer2 wins
pub fn determine_winner(results1: &EvaluationResults, results2: &EvaluationResults) -> i32 {
    let mut score1 = 0;
    let mut score2 = 0;

    // Tier 1: Most important (weight = 3)
    if results1.renyi_entropy > results2.renyi_entropy {
        score1 += 3;
    } else {
        score2 += 3;
    }

    if results1.fertility < results2.fertility {
        score1 += 3;
    } else {
        score2 += 3;
    }

    // Tier 2: Standard metrics (weight = 2)
    if results1.compression_rate > results2.compression_rate {
        score1 += 2;
    } else {
        score2 += 2;
    }

    if results1.vocab_utilization > results2.vocab_utilization {
        score1 += 2;
    } else {
        score2 += 2;
    }

    if results1.oov_rate < results2.oov_rate {
        score1 += 2;
    } else {
        score2 += 2;
    }

    if results1.encoding_speed_chars_per_sec > results2.encoding_speed_chars_per_sec {
        score1 += 2;
    } else {
        score2 += 2;
    }

    // Tier 3: Malayalam-specific (weight = 3)
    if results1.morphological_consistency > results2.morphological_consistency {
        score1 += 3;
    } else {
        score2 += 3;
    }

    if results1.rare_word_handling_ratio < results2.rare_word_handling_ratio {
        score1 += 3;
    } else {
        score2 += 3;
    }

    if results1.shannon_entropy > results2.shannon_entropy {
        score1 += 2;
    } else {
        score2 += 2;
    }

    score1 - score2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renyi_entropy_uniform_distribution() {
        let mut freqs = HashMap::new();
        // Uniform distribution: 100 tokens appearing once each
        for i in 0..100 {
            freqs.insert(i, 1);
        }
        let entropy = calculate_renyi_entropy(&freqs, 100);
        // For uniform distribution, entropy should be high
        assert!(entropy > 5.0);
    }

    #[test]
    fn test_renyi_entropy_concentrated_distribution() {
        let mut freqs = HashMap::new();
        // Concentrated: one token appears 100 times
        freqs.insert(0, 100);
        let entropy = calculate_renyi_entropy(&freqs, 100);
        // For concentrated distribution, entropy should be low
        assert!(entropy < 1.0);
    }

    #[test]
    fn test_shannon_entropy_calculation() {
        let mut freqs = HashMap::new();
        freqs.insert(0, 50);
        freqs.insert(1, 50);
        let entropy = calculate_shannon_entropy(&freqs, 100);
        // Two equally probable events: entropy should be 1.0
        assert!((entropy - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_count_words() {
        let text = "hello world this is test";
        assert_eq!(count_words(text), 5);
    }

    #[test]
    fn test_count_words_with_newlines() {
        let text = "hello\nworld\ntest";
        assert_eq!(count_words(text), 3);
    }

    #[test]
    fn test_count_words_empty() {
        assert_eq!(count_words(""), 0);
    }

    #[test]
    fn test_determine_winner_tokenizer1_better() {
        let results1 = EvaluationResults {
            renyi_entropy: 10.5,
            fertility: 2.0,
            compression_rate: 5.0,
            vocab_utilization: 70.0,
            oov_rate: 0.05,
            encoding_speed_chars_per_sec: 40000.0,
            morphological_consistency: 85.0,
            rare_word_handling_ratio: 1.2,
            shannon_entropy: 10.0,
            total_tokens: 1000,
            unique_tokens: 500,
            total_characters: 5000,
            total_words: 500,
            vocab_size: 16000,
        };

        let results2 = EvaluationResults {
            renyi_entropy: 10.0,
            fertility: 2.5,
            compression_rate: 4.5,
            vocab_utilization: 65.0,
            oov_rate: 0.1,
            encoding_speed_chars_per_sec: 35000.0,
            morphological_consistency: 75.0,
            rare_word_handling_ratio: 1.5,
            shannon_entropy: 9.5,
            total_tokens: 1100,
            unique_tokens: 450,
            total_characters: 5000,
            total_words: 500,
            vocab_size: 16000,
        };

        let winner = determine_winner(&results1, &results2);
        assert!(winner > 0, "Tokenizer 1 should win");
    }

    #[test]
    fn test_token_stats_collection() {
        let tokens = vec![1, 2, 2, 3, 3, 3];
        let stats = collect_token_stats(&tokens);

        assert_eq!(stats.total_tokens, 6);
        assert_eq!(stats.unique_tokens, 3);
        assert_eq!(*stats.token_frequencies.get(&1).unwrap(), 1);
        assert_eq!(*stats.token_frequencies.get(&2).unwrap(), 2);
        assert_eq!(*stats.token_frequencies.get(&3).unwrap(), 3);
    }
}
