use candle_core::{DType, Tensor};
use mospeada::{repo::Repo, Result};
use tokenizers;
fn main() -> Result<()> {
    println!("mospeada-cross-encoder");

    let device = mospeada::gpu(0)?;

    let repo =
        mospeada::hf_hub::from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2", None, None, None)?;
    let sentences = vec![
        (
            "How many people live in Berlin?",
            "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        ),
        (
            "How many people live in Berlin?",
            "New York City is famous for the Metropolitan Museum of Art.",
        ),
    ];

    let tokenizer = {
        let mut tokenizer = repo.load_tokenizer()?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams::default()));

        tokenizer
    };

    let bert = repo.load_model(DType::F32, &device, |config, vb| {
        candle_transformers::models::bert::BertForSequenceClassification::load(vb, config)
    })?;

    let encoded = tokenizer.encode_batch(sentences, true)?;

    let ids = encoded
        .iter()
        .map(|e| Tensor::new(e.get_ids(), &device).unwrap())
        .collect::<Vec<_>>();
    let ids = Tensor::stack(&ids, 0)?;
    let type_ids = encoded
        .iter()
        .map(|e| Tensor::new(e.get_type_ids(), &device).unwrap())
        .collect::<Vec<_>>();
    let type_ids = Tensor::stack(&type_ids, 0)?;
    let attention_mask = encoded
        .iter()
        .map(|e| Tensor::new(e.get_attention_mask(), &device).unwrap())
        .collect::<Vec<_>>();
    let attention_mask = Tensor::stack(&attention_mask, 0)?;

    let result = bert.forward(&ids, &type_ids, Some(&attention_mask))?;

    mospeada::debug::print_tensor::<f32>(&result)?;

    Ok(())
}
