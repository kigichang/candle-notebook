// use tokenizers::models::bpe::BPE;
// use tokenizers::tokenizer::{Tokenizer, AddedToken, Encoding};
// use tokenizers::utils::truncate_sequences;
// use std::fs::File;
// use std::path::Path;
// use std::io::BufReader;
// use serde::Deserialize;

// // Define a struct to parse the special tokens map
// #[derive(Deserialize, Debug)]
// struct SpecialTokensMap {
//     cls_token: String,
//     sep_token: String,
//     pad_token: String,
//     unk_token: String,
//     mask_token: Option<String>, // Optional, depending on your tokenizer
// }

// fn main() -> Result<(), Box<dyn std::error::Error>> {
//     // Paths to files
//     let vocab_file = "path/to/vocab.txt";
//     let special_tokens_map_file = "path/to/special_tokens_map.json";

//     // Load the vocabulary and special tokens
//     let vocab = BPE::from_files(vocab_file)?.build()?;
//     let mut tokenizer = Tokenizer::new(vocab);

//     // Load special tokens
//     let file = File::open(special_tokens_map_file)?;
//     let reader = BufReader::new(file);
//     let special_tokens: SpecialTokensMap = serde_json::from_reader(reader)?;

//     // Add special tokens
//     tokenizer.add_special_tokens(&[
//         AddedToken::from(special_tokens.cls_token, true),
//         AddedToken::from(special_tokens.sep_token, true),
//         AddedToken::from(special_tokens.pad_token, true),
//         AddedToken::from(special_tokens.unk_token, true),
//     ]);

//     if let Some(mask_token) = special_tokens.mask_token {
//         tokenizer.add_special_tokens(&[AddedToken::from(mask_token, true)]);
//     }

//     // Optional: Add additional configuration from tokenizer_config.json
//     let tokenizer_config_file = "path/to/tokenizer_config.json";
//     if Path::new(tokenizer_config_file).exists() {
//         let file = File::open(tokenizer_config_file)?;
//         let reader = BufReader::new(file);
//         let config: serde_json::Value = serde_json::from_reader(reader)?;
//         println!("Loaded tokenizer config: {:?}", config);
//     }

//     // Encode some text
//     let encoding: Encoding = tokenizer.encode("Hello, how are you?", true)?;
//     println!("{:?}", encoding.get_tokens());

//     Ok(())
// }

use anyhow::{Error as E, Result};
use serde::Deserialize;
use tokenizers::decoders::bpe;
use tokenizers::models::bpe::{BpeBuilder, BPE};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::tokenizer::AddedToken;
use tokenizers::{Tokenizer, TokenizerBuilder};

// Define a struct to parse the special tokens map
#[derive(Deserialize, Debug)]
struct SpecialTokensMap {
    cls_token: String,
    sep_token: String,
    pad_token: String,
    unk_token: String,
    mask_token: Option<String>, // Optional, depending on your tokenizer
}

#[test]
fn load_tokenizer() -> Result<()> {
    let vocab_file = "/Users/kigi/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/b2cfda50a1a9fc7919e7444afbb52610d268af92/vocab.txt";
    let special_tokens_map_file = "/Users/kigi/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/b2cfda50a1a9fc7919e7444afbb52610d268af92/special_tokens_map.json";
    let tokenizer_config_file = "/Users/kigi/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/b2cfda50a1a9fc7919e7444afbb52610d268af92/tokenizer_config.json";

    let word_piece_builder = WordPiece::from_file(vocab_file);
    let word_piece = word_piece_builder
        .unk_token("[UNK]".to_owned())
        .build()
        .map_err(E::msg)?;

    let mut tokenizer = Tokenizer::new(word_piece);

    let special_tokens: SpecialTokensMap =
        serde_json::from_reader(std::fs::File::open(special_tokens_map_file)?)?;

    println!("{:?}", special_tokens);
    let added = tokenizer.add_special_tokens(&[
        AddedToken::from(special_tokens.cls_token, true),
        AddedToken::from(special_tokens.sep_token, true),
        AddedToken::from(special_tokens.pad_token, true),
        AddedToken::from(special_tokens.unk_token, true),
    ]);
    println!("{:?}", added);

    if let Some(mask_token) = special_tokens.mask_token {
        tokenizer.add_special_tokens(&[AddedToken::from(mask_token, true)]);
    }

    println!("{:?}", tokenizer.get_vocab_size(false));
    println!("{:?}", tokenizer.get_vocab_size(true));

    let tokenizer = tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));

    let tokenizer = tokenizer
        .with_truncation(Some(tokenizers::TruncationParams::default()))
        .map_err(E::msg)?;

    let encoding = tokenizer
        .encode("Hello, how are you?", true)
        .map_err(E::msg)?;
    println!("{:?}", encoding.get_ids());

    Ok(())
}
