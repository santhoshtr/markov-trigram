use anyhow::{Result, anyhow};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

const BPE_REPO: &str = "smcproject/malayalam-bpe-tokenizer";
const UNIGRAM_REPO: &str = "smcproject/malayalam-unigram-tokenizer";

#[derive(Clone, Debug, Default, clap::ValueEnum)]
pub enum TokenizerType {
    #[default]
    Unigram,
    Bpe,
}

/// Load a tokenizer from a local path or from HuggingFace Hub.
///
/// If `local_path` is Some, the file is loaded directly.
/// Otherwise, the tokenizer is fetched from HF Hub (cached in ~/.cache/huggingface/).
pub fn load_tokenizer(
    local_path: Option<&str>,
    tokenizer_type: &TokenizerType,
) -> Result<Tokenizer> {
    let path = match local_path {
        Some(p) => p.to_string(),
        None => {
            let repo_id = match tokenizer_type {
                TokenizerType::Bpe => BPE_REPO,
                TokenizerType::Unigram => UNIGRAM_REPO,
            };
            eprintln!("Fetching tokenizer from HuggingFace Hub: {repo_id}");
            let api = Api::new().map_err(|e| anyhow!("Failed to create HF Hub API: {e}"))?;
            let repo = api.model(repo_id.to_string());
            repo.get("tokenizer.json")
                .map_err(|e| anyhow!("Failed to download tokenizer from {repo_id}: {e}"))?
                .to_str()
                .ok_or_else(|| anyhow!("Tokenizer path is not valid UTF-8"))?
                .to_string()
        }
    };

    Tokenizer::from_file(&path).map_err(|e| anyhow!("Failed to load tokenizer from {path}: {e}"))
}
