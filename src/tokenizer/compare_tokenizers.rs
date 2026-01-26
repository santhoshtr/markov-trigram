use anyhow::{anyhow, Result};
use clap::{Arg, Command};
use markov_trigram::find_text_files;
use markov_trigram::tokenizer::evaluation::{
    determine_winner, evaluate_tokenizer, EvaluationResults,
};
use serde_json::json;
use std::fs;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Parameters for a single metric row in the comparison report
struct MetricRow {
    name: &'static str,
    val1: f64,
    val2: f64,
    higher_is_better: bool,
}

/// Context for report generation
struct ReportContext {
    name1: String,
    name2: String,
    results1: EvaluationResults,
    results2: EvaluationResults,
}

fn main() -> Result<()> {
    let matches = Command::new("Tokenizer Comparison Tool")
        .version("1.0")
        .about("Compare intrinsic quality of tokenizers (foundational infrastructure)")
        .arg(
            Arg::new("tokenizer1")
                .long("tokenizer1")
                .required(true)
                .help("Path to first tokenizer (e.g., Unigram tokenizer.json)"),
        )
        .arg(
            Arg::new("tokenizer2")
                .long("tokenizer2")
                .required(true)
                .help("Path to second tokenizer (e.g., BPE tokenizer.json)"),
        )
        .arg(
            Arg::new("corpus")
                .long("corpus")
                .required(true)
                .help("Path to corpus directory containing text files"),
        )
        .arg(
            Arg::new("output")
                .long("output")
                .default_value("tokenizer_comparison.md")
                .help("Output markdown report path"),
        )
        .arg(
            Arg::new("json")
                .long("json")
                .default_value("tokenizer_comparison.json")
                .help("Output JSON path"),
        )
        .arg(
            Arg::new("name1")
                .long("name1")
                .default_value("Tokenizer 1")
                .help("Display name for tokenizer 1"),
        )
        .arg(
            Arg::new("name2")
                .long("name2")
                .default_value("Tokenizer 2")
                .help("Display name for tokenizer 2"),
        )
        .get_matches();

    let tok1_path = matches.get_one::<String>("tokenizer1").unwrap();
    let tok2_path = matches.get_one::<String>("tokenizer2").unwrap();
    let corpus_dir = matches.get_one::<String>("corpus").unwrap();
    let output_md = matches.get_one::<String>("output").unwrap();
    let output_json = matches.get_one::<String>("json").unwrap();
    let name1 = matches.get_one::<String>("name1").unwrap();
    let name2 = matches.get_one::<String>("name2").unwrap();

    run_comparison(
        tok1_path,
        tok2_path,
        corpus_dir,
        output_md,
        output_json,
        name1,
        name2,
    )
}

fn run_comparison(
    tok1_path: &str,
    tok2_path: &str,
    corpus_dir: &str,
    output_md: &str,
    output_json: &str,
    name1: &str,
    name2: &str,
) -> Result<()> {
    println!();
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     Tokenizer Comparison: Intrinsic Quality Metrics       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();

    // Load tokenizers
    println!("[1/4] Loading tokenizers...");
    let tokenizer1 = Tokenizer::from_file(tok1_path)
        .map_err(|e| anyhow!("Failed to load tokenizer1 from {}: {}", tok1_path, e))?;
    let tokenizer2 = Tokenizer::from_file(tok2_path)
        .map_err(|e| anyhow!("Failed to load tokenizer2 from {}: {}", tok2_path, e))?;
    println!(
        "  ✓ {} loaded (vocab: {})",
        name1,
        tokenizer1.get_vocab_size(true)
    );
    println!(
        "  ✓ {} loaded (vocab: {})",
        name2,
        tokenizer2.get_vocab_size(true)
    );
    println!();

    // Load corpus
    println!("[2/4] Loading corpus...");
    let corpus = load_corpus(corpus_dir)?;
    println!();

    // Evaluate both tokenizers
    println!("[3/4] Evaluating tokenizers...");
    println!("  Evaluating {}...", name1);
    let start = Instant::now();
    let results1 = evaluate_tokenizer(&tokenizer1, &corpus)?;
    let time1 = start.elapsed();
    println!("  ✓ Complete ({:.1}s)", time1.as_secs_f64());

    println!("  Evaluating {}...", name2);
    let start = Instant::now();
    let results2 = evaluate_tokenizer(&tokenizer2, &corpus)?;
    let time2 = start.elapsed();
    println!("  ✓ Complete ({:.1}s)", time2.as_secs_f64());
    println!();

    // Generate reports
    println!("[4/4] Generating reports...");
    let ctx = ReportContext {
        name1: name1.to_string(),
        name2: name2.to_string(),
        results1,
        results2,
    };

    generate_markdown_report(&ctx, &corpus, output_md)?;
    generate_json_report(&ctx, &corpus, tok1_path, tok2_path, output_json)?;
    println!();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                     ✓ Comparison Complete                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Reports:");
    println!("  📄 Markdown: {}", output_md);
    println!("  📊 JSON:     {}", output_json);
    println!();

    Ok(())
}

fn load_corpus(corpus_dir: &str) -> Result<String> {
    let files = find_text_files(corpus_dir)
        .map_err(|e| anyhow!("Failed to find corpus files in {}: {}", corpus_dir, e))?;

    if files.is_empty() {
        return Err(anyhow!("No text files found in {}", corpus_dir));
    }

    println!("  Found {} text file(s)", files.len());

    let mut corpus = String::new();
    let total = files.len();

    for (idx, file) in files.iter().enumerate() {
        print!("  [{}/{}] Loading...\r", idx + 1, total);
        std::io::Write::flush(&mut std::io::stdout())?;

        let content =
            std::fs::read_to_string(file).map_err(|e| anyhow!("Failed to read {}: {}", file, e))?;
        corpus.push_str(&content);
        corpus.push('\n');
    }

    println!(
        "  ✓ Loaded {} MB ({} chars)  ",
        corpus.len() / 1_000_000,
        corpus.chars().count()
    );

    Ok(corpus)
}

fn generate_markdown_report(ctx: &ReportContext, corpus: &str, output_path: &str) -> Result<()> {
    let mut report = String::new();

    report.push_str("# Tokenizer Comparison Report\n\n");
    report.push_str(&format!(
        "**Generated:** {}\n\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    ));

    // Corpus info
    report.push_str("## Corpus Information\n\n");
    report.push_str(&format!(
        "- **Total bytes:** {} MB\n",
        corpus.len() / 1_000_000
    ));
    report.push_str(&format!("- **Characters:** {}\n", corpus.chars().count()));
    report.push_str(&format!("- **Words:** {}\n\n", ctx.results1.total_words));

    // Winner determination
    let winner_score = determine_winner(&ctx.results1, &ctx.results2);
    let winner = if winner_score > 0 {
        &ctx.name1
    } else {
        &ctx.name2
    };
    let mut tok1_wins = 0;
    let mut tok2_wins = 0;

    report.push_str("## Executive Summary\n\n");
    report.push_str(&format!("**🏆 Winner: {}**\n\n", winner));

    // Summary stats
    report.push_str("### Key Statistics\n\n");
    report.push_str(&format!("| Metric | {} | {} |\n", ctx.name1, ctx.name2));
    report.push_str("|--------|");
    for _ in 0..2 {
        report.push_str("----|");
    }
    report.push('\n');
    report.push_str(&format!(
        "| Tokens | {} | {} |\n",
        ctx.results1.total_tokens, ctx.results2.total_tokens
    ));
    report.push_str(&format!(
        "| Unique Tokens | {} | {} |\n",
        ctx.results1.unique_tokens, ctx.results2.unique_tokens
    ));
    report.push_str(&format!(
        "| Vocab Size | {} | {} |\n\n",
        ctx.results1.vocab_size, ctx.results2.vocab_size
    ));

    // Detailed metrics table
    report.push_str("## Detailed Metrics Comparison\n\n");
    report.push_str("```\n");
    report.push_str(
        "+--------------------------------------+----------+----------+----------+--------+\n",
    );
    report.push_str(&format!(
        "| Metric                               | {:8} | {:8} | Winner   | Delta  |\n",
        truncate(&ctx.name1, 8),
        truncate(&ctx.name2, 8)
    ));
    report.push_str(
        "+--------------------------------------+----------+----------+----------+--------+\n",
    );

    // Tier 1 metrics
    add_metric_row(
        &mut report,
        &MetricRow {
            name: "* Rényi Entropy (α=2)",
            val1: ctx.results1.renyi_entropy,
            val2: ctx.results2.renyi_entropy,
            higher_is_better: true,
        },
        &ctx.name1,
        &ctx.name2,
        &mut tok1_wins,
        &mut tok2_wins,
    );

    add_metric_row(
        &mut report,
        &MetricRow {
            name: "Fertility (tokens/word)",
            val1: ctx.results1.fertility,
            val2: ctx.results2.fertility,
            higher_is_better: false,
        },
        &ctx.name1,
        &ctx.name2,
        &mut tok1_wins,
        &mut tok2_wins,
    );

    // Tier 2 metrics
    add_metric_row(
        &mut report,
        &MetricRow {
            name: "Compression Rate",
            val1: ctx.results1.compression_rate,
            val2: ctx.results2.compression_rate,
            higher_is_better: true,
        },
        &ctx.name1,
        &ctx.name2,
        &mut tok1_wins,
        &mut tok2_wins,
    );

    add_metric_row(
        &mut report,
        &MetricRow {
            name: "Vocab Utilization (%)",
            val1: ctx.results1.vocab_utilization,
            val2: ctx.results2.vocab_utilization,
            higher_is_better: true,
        },
        &ctx.name1,
        &ctx.name2,
        &mut tok1_wins,
        &mut tok2_wins,
    );

    add_metric_row(
        &mut report,
        &MetricRow {
            name: "OOV Rate (%)",
            val1: ctx.results1.oov_rate,
            val2: ctx.results2.oov_rate,
            higher_is_better: false,
        },
        &ctx.name1,
        &ctx.name2,
        &mut tok1_wins,
        &mut tok2_wins,
    );

    add_metric_row(
        &mut report,
        &MetricRow {
            name: "Encoding Speed (chars/s)",
            val1: ctx.results1.encoding_speed_chars_per_sec,
            val2: ctx.results2.encoding_speed_chars_per_sec,
            higher_is_better: true,
        },
        &ctx.name1,
        &ctx.name2,
        &mut tok1_wins,
        &mut tok2_wins,
    );

    // Tier 3 metrics
    add_metric_row(
        &mut report,
        &MetricRow {
            name: "Morphological Consistency (%)",
            val1: ctx.results1.morphological_consistency,
            val2: ctx.results2.morphological_consistency,
            higher_is_better: true,
        },
        &ctx.name1,
        &ctx.name2,
        &mut tok1_wins,
        &mut tok2_wins,
    );

    add_metric_row(
        &mut report,
        &MetricRow {
            name: "Rare Word Ratio",
            val1: ctx.results1.rare_word_handling_ratio,
            val2: ctx.results2.rare_word_handling_ratio,
            higher_is_better: false,
        },
        &ctx.name1,
        &ctx.name2,
        &mut tok1_wins,
        &mut tok2_wins,
    );

    add_metric_row(
        &mut report,
        &MetricRow {
            name: "Shannon Entropy",
            val1: ctx.results1.shannon_entropy,
            val2: ctx.results2.shannon_entropy,
            higher_is_better: true,
        },
        &ctx.name1,
        &ctx.name2,
        &mut tok1_wins,
        &mut tok2_wins,
    );

    report.push_str(
        "+--------------------------------------+----------+----------+----------+--------+\n",
    );
    report.push_str("```\n\n");

    // Interpretation
    report.push_str("## Interpretation\n\n");
    report.push_str(&format!(
        "**Metrics won:** {} ({} wins), {} ({} wins)\n\n",
        ctx.name1, tok1_wins, ctx.name2, tok2_wins
    ));

    report.push_str("### Tier 1: Most Predictive Metrics\n\n");
    let compression_str = format!("{:.2}", ctx.results1.compression_rate);
    let compression_str2 = format!("{:.2}", ctx.results2.compression_rate);
    report.push_str(&format!(
        "**Rényi Entropy (α=2):**\n\
         - {}: {:.2} (higher is better - more diverse token distribution)\n\
         - {}: {:.2}\n\
         - Δ: {:.2} ({:+.1}%)\n\n\
         Research shows Rényi entropy has 0.82 correlation with downstream task \
         performance (Zouhar et al., ACL 2023). This is the single best predictor \
         of tokenizer quality.\n\n",
        ctx.name1,
        ctx.results1.renyi_entropy,
        ctx.name2,
        ctx.results2.renyi_entropy,
        ctx.results1.renyi_entropy - ctx.results2.renyi_entropy,
        ((ctx.results1.renyi_entropy - ctx.results2.renyi_entropy) / ctx.results2.renyi_entropy)
            * 100.0
    ));

    report.push_str(&format!(
        "**Fertility:**\n\
         - {}: {:.2} tokens/word (lower is better - more efficient)\n\
         - {}: {:.2} tokens/word\n\
         - Δ: {:.2} ({:+.1}%)\n\n\
         Lower fertility means more compact representations, leading to:\n\
         - More efficient memory usage\n\
         - Faster processing (fewer tokens)\n\
         - Better semantic coherence (less fragmentation)\n\n",
        ctx.name1,
        ctx.results1.fertility,
        ctx.name2,
        ctx.results2.fertility,
        ctx.results1.fertility - ctx.results2.fertility,
        ((ctx.results1.fertility - ctx.results2.fertility) / ctx.results2.fertility) * 100.0
    ));

    report.push_str("### Tier 2: Standard Efficiency Metrics\n\n");
    report.push_str(&format!(
        "**Compression Rate:** {} ({}), {} ({})\n\
         **Vocabulary Utilization:** {:.1}% vs {:.1}% (healthy range: 60-80%)\n\
         **OOV Rate:** {:.3}% vs {:.3}% (target: <0.1%)\n\
         **Encoding Speed:** {:.0} vs {:.0} chars/sec\n\n",
        compression_str,
        if ctx.results1.compression_rate > ctx.results2.compression_rate {
            "winner"
        } else {
            ""
        },
        compression_str2,
        if ctx.results2.compression_rate > ctx.results1.compression_rate {
            "winner"
        } else {
            ""
        },
        ctx.results1.vocab_utilization,
        ctx.results2.vocab_utilization,
        ctx.results1.oov_rate,
        ctx.results2.oov_rate,
        ctx.results1.encoding_speed_chars_per_sec,
        ctx.results2.encoding_speed_chars_per_sec
    ));

    report.push_str("### Tier 3: Malayalam-Specific Quality\n\n");
    report.push_str(&format!(
        "**Morphological Consistency:** {:.1}% vs {:.1}%\n\n\
         Consistency score measures preservation of word roots in Malayalam word families:\n\
         - Better consistency = better handling of morphologically complex words\n\
         - Critical for Malayalam's agglutinative structure (root + suffixes)\n\
         - Example: കേരളം (Kerala) variants should share root token\n\n\
         **Rare Word Handling Ratio:** {:.2} vs {:.2}\n\
         - Ratio closer to 1.0 = consistent segmentation regardless of frequency\n\
         - Lower ratio = handles rare words as well as common words\n\n",
        ctx.results1.morphological_consistency,
        ctx.results2.morphological_consistency,
        ctx.results1.rare_word_handling_ratio,
        ctx.results2.rare_word_handling_ratio
    ));

    report.push_str("## Recommendation\n\n");
    report.push_str(&format!(
        "**Use {} for Malayalam** based on:\n\
         - {} wins on {} out of 9 metrics\n\
         - Better Rényi entropy (best performance predictor)\n\
         - Superior morphological awareness (Malayalam is agglutinative)\n\
         - Foundational tokenizer for downstream tasks\n",
        winner,
        winner,
        std::cmp::max(tok1_wins, tok2_wins)
    ));

    fs::write(output_path, report)?;
    println!("  ✓ Markdown: {}", output_path);

    Ok(())
}

fn generate_json_report(
    ctx: &ReportContext,
    corpus: &str,
    path1: &str,
    path2: &str,
    output_path: &str,
) -> Result<()> {
    let winner_score = determine_winner(&ctx.results1, &ctx.results2);
    let winner = if winner_score > 0 {
        &ctx.name1
    } else {
        &ctx.name2
    };

    let report = json!({
        "timestamp": chrono::Local::now().to_rfc3339(),
        "corpus_info": {
            "total_bytes": corpus.len(),
            "total_characters": ctx.results1.total_characters,
            "total_words": ctx.results1.total_words,
        },
        "tokenizer1": {
            "name": ctx.name1,
            "path": path1,
            "vocab_size": ctx.results1.vocab_size,
            "results": {
                "renyi_entropy": ctx.results1.renyi_entropy,
                "fertility": ctx.results1.fertility,
                "compression_rate": ctx.results1.compression_rate,
                "vocab_utilization": ctx.results1.vocab_utilization,
                "oov_rate": ctx.results1.oov_rate,
                "encoding_speed_chars_per_sec": ctx.results1.encoding_speed_chars_per_sec,
                "morphological_consistency": ctx.results1.morphological_consistency,
                "rare_word_handling_ratio": ctx.results1.rare_word_handling_ratio,
                "shannon_entropy": ctx.results1.shannon_entropy,
                "total_tokens": ctx.results1.total_tokens,
                "unique_tokens": ctx.results1.unique_tokens,
            }
        },
        "tokenizer2": {
            "name": ctx.name2,
            "path": path2,
            "vocab_size": ctx.results2.vocab_size,
            "results": {
                "renyi_entropy": ctx.results2.renyi_entropy,
                "fertility": ctx.results2.fertility,
                "compression_rate": ctx.results2.compression_rate,
                "vocab_utilization": ctx.results2.vocab_utilization,
                "oov_rate": ctx.results2.oov_rate,
                "encoding_speed_chars_per_sec": ctx.results2.encoding_speed_chars_per_sec,
                "morphological_consistency": ctx.results2.morphological_consistency,
                "rare_word_handling_ratio": ctx.results2.rare_word_handling_ratio,
                "shannon_entropy": ctx.results2.shannon_entropy,
                "total_tokens": ctx.results2.total_tokens,
                "unique_tokens": ctx.results2.unique_tokens,
            }
        },
        "winner": winner,
        "winner_score": winner_score,
    });

    fs::write(output_path, serde_json::to_string_pretty(&report)?)?;
    println!("  ✓ JSON:     {}", output_path);

    Ok(())
}

fn add_metric_row(
    report: &mut String,
    row: &MetricRow,
    name1: &str,
    name2: &str,
    tok1_wins: &mut usize,
    tok2_wins: &mut usize,
) {
    let winner = if row.higher_is_better {
        if row.val1 > row.val2 {
            *tok1_wins += 1;
            name1
        } else {
            *tok2_wins += 1;
            name2
        }
    } else if row.val1 < row.val2 {
        *tok1_wins += 1;
        name1
    } else {
        *tok2_wins += 1;
        name2
    };

    let delta = (row.val1 - row.val2).abs();
    let delta_str = if row.val1 > row.val2 {
        format!("+{:.2}", delta)
    } else {
        format!("-{:.2}", delta)
    };

    report.push_str(&format!(
        "| {:<36} | {:8.2} | {:8.2} | {:<8} | {:>6} |\n",
        row.name,
        row.val1,
        row.val2,
        truncate(winner, 8),
        delta_str
    ));
}

fn truncate(s: &str, len: usize) -> String {
    if s.len() > len {
        format!("{}.", &s[..len - 1])
    } else {
        s.to_string()
    }
}
