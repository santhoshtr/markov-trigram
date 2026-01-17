use clap::{Arg, Command};
use markov_trigram::find_text_files;
use tokenizers::decoders::metaspace::{Metaspace, PrependScheme};
use tokenizers::models::unigram::{Unigram, UnigramTrainerBuilder};
use tokenizers::normalizers::{strip::Strip, unicode::NFC, utils::Sequence};
use tokenizers::pre_tokenizers::metaspace::Metaspace as MetaspacePreTokenizer;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::AddedToken;
use tokenizers::{Result, Tokenizer, TokenizerBuilder};

fn train_tokenizer(folder_path: &str, vocab_size: usize, output_path: &str) -> Result<()> {
    let mut trainer = UnigramTrainerBuilder::default()
        .show_progress(true)
        .vocab_size(vocab_size as u32)
        .unk_token(Some(String::from("<unk>")))
        .special_tokens(vec![
            AddedToken::from(String::from("<s>"), true),
            AddedToken::from(String::from("<pad>"), true),
            AddedToken::from(String::from("</s>"), true),
            AddedToken::from(String::from("<unk>"), true),
            AddedToken::from(String::from("<mask>"), true),
        ])
        .build()
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

    // Consistent Unigram setup with Metaspace for Malayalam
    let mut tokenizer: tokenizers::TokenizerImpl<Unigram, Sequence, Metaspace, _, Metaspace> =
        TokenizerBuilder::new()
            .with_model(Unigram::default())
            .with_normalizer(Some(Sequence::new(vec![
                Strip::new(true, true).into(),
                NFC.into(),
            ])))
            .with_pre_tokenizer(Some(MetaspacePreTokenizer::new(
                '▁',                   // replacement characteor
                PrependScheme::Always, // prepend_scheme: add ▁ at the beginning
                true,                  // prepend_scheme: add ▁ at the beginning
            )))
            .with_decoder(Some(Metaspace::new(
                '▁',
                PrependScheme::Always,
                true, // prepend_scheme (must match pre-tokenizer)
            )))
            .with_post_processor(Some(
                TemplateProcessing::builder()
                    .try_single("<s> $A </s>")
                    .unwrap()
                    .try_pair("<s> $A </s> $B:1 </s>:1")
                    .unwrap()
                    .special_tokens(vec![("<s>", 0), ("</s>", 2)])
                    .build()
                    .unwrap(),
            ))
            .build()?;

    let files = find_text_files(folder_path)?;

    if files.is_empty() {
        eprintln!("No *.txt files found in {}", folder_path);
        std::process::exit(1);
    }

    println!("Found {} text files:", files.len());

    tokenizer.train_from_files(&mut trainer, files)?;
    tokenizer.save(output_path, true)?;

    println!("Tokenizer saved to: {}", output_path);

    // Test encoding with Malayalam text
    let test_texts = vec![
        "Hello, y'all! How are you 😁 ?",
        "നമസ്കാരം",    // Malayalam: Hello
        "എങ്ങനെയുണ്ട്?", // Malayalam: How are you?
    ];

    for text in test_texts {
        let output = tokenizer.encode(text, true)?;
        println!("\nTest text: {}", text);
        println!("Tokens: {:?}", output.get_tokens());
        println!("Token IDs: {:?}", output.get_ids());
    }

    Ok(())
}

fn encode_text(tokenizer_path: &str, text: &str) -> Result<()> {
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    let encoded = tokenizer.encode(text, true)?;

    println!("Input text: {}", text);
    println!("Tokens: {:?}", encoded.get_tokens());
    println!("Token IDs: {:?}", encoded.get_ids());

    Ok(())
}

fn main() -> Result<()> {
    let matches = Command::new("ML Tokenizer")
        .version("1.0")
        .author("Your Name")
        .about("Train and use Unigram tokenizers for Malayalam")
        .subcommand(
            Command::new("train")
                .about("Train a new tokenizer")
                .arg(
                    Arg::new("folder")
                        .short('f')
                        .long("folder")
                        .value_name("FOLDER")
                        .help("Folder containing training text files")
                        .required(true),
                )
                .arg(
                    Arg::new("vocab-size")
                        .short('v')
                        .long("vocab-size")
                        .value_name("SIZE")
                        .help("Vocabulary size")
                        .default_value("16000"),
                )
                .arg(
                    Arg::new("output")
                        .short('o')
                        .long("output")
                        .value_name("FILE")
                        .help("Output tokenizer file")
                        .default_value("data/tokenizer.ml.json"),
                ),
        )
        .subcommand(
            Command::new("encode")
                .about("Encode text using a trained tokenizer")
                .arg(
                    Arg::new("tokenizer")
                        .short('t')
                        .long("tokenizer")
                        .value_name("FILE")
                        .help("Path to tokenizer file")
                        .default_value("data/tokenizer.ml.json"),
                )
                .arg(
                    Arg::new("text")
                        .value_name("TEXT")
                        .help("Text to encode")
                        .required(true),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("train", sub_matches)) => {
            let folder_path = sub_matches.get_one::<String>("folder").unwrap();
            let vocab_size: usize = sub_matches
                .get_one::<String>("vocab-size")
                .unwrap()
                .parse()
                .expect("Invalid vocabulary size");
            let output_path = sub_matches.get_one::<String>("output").unwrap();

            train_tokenizer(folder_path, vocab_size, output_path)?;
        }
        Some(("encode", sub_matches)) => {
            let tokenizer_path = sub_matches.get_one::<String>("tokenizer").unwrap();
            let text = sub_matches.get_one::<String>("text").unwrap();

            encode_text(tokenizer_path, text)?;
        }
        _ => {
            eprintln!("Please specify a subcommand. Use --help for more information.");
            std::process::exit(1);
        }
    }

    Ok(())
}
