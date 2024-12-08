use anyhow::Result;
use candle_core::Tensor;
use macross::{AutoModel, AutoTokenizer};

fn main() -> Result<()> {
    let device = macross::device(false)?;
    let sentences = vec!["This is an example sentence", "Each sentence is converted"];
    let tokenizer = {
        let mut tokenizer =
            AutoTokenizer::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                .map_err(anyhow::Error::msg)?;
        let params = tokenizers::PaddingParams::default();
        let truncation = tokenizers::TruncationParams::default();
        let tokenizer = tokenizer.with_padding(Some(params));
        let tokenizer = tokenizer
            .with_truncation(Some(truncation))
            .map_err(anyhow::Error::msg)?;
        tokenizer.clone()
    };

    let bert = macross::models::bert::BertModel::from_pretrained(
        ("sentence-transformers/all-MiniLM-L6-v2", true),
        candle_core::DType::F32,
        &device,
    )?;

    let encoded_input = tokenizer
        .encode_batch(sentences, true)
        .map_err(anyhow::Error::msg)?;

    let ids = encoded_input
        .iter()
        .map(|e| Tensor::new(e.get_ids(), &device).unwrap())
        .collect::<Vec<_>>();

    let ids = Tensor::stack(&ids, 0)?;
    let type_ids = encoded_input
        .iter()
        .map(|e| Tensor::new(e.get_type_ids(), &device).unwrap())
        .collect::<Vec<_>>();
    let type_ids = Tensor::stack(&type_ids, 0)?;
    let attention_mask = encoded_input
        .iter()
        .map(|e| Tensor::new(e.get_attention_mask(), &device).unwrap())
        .collect::<Vec<_>>();
    let attention_mask = Tensor::stack(&attention_mask, 0)?;

    let result = bert.forward(&ids, &type_ids, Some(&attention_mask))?;
    let mean = macross::models::bert::mean_pooling(&result.0, &attention_mask)?;
    let result = macross::normalize(&mean)?;
    println!("result: {:?}", result.to_vec2::<f32>()?);
    Ok(())
}
