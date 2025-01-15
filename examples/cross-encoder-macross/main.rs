use anyhow::{Error as E, Result};
use candle_core::{DType, Tensor};
use macross::{models::bert::BertForSequenceClassification, AutoModel, AutoTokenizer};

fn main() -> Result<()> {
    let device = macross::device(false)?;
    let bert = BertForSequenceClassification::from_pretrained(
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", true),
        DType::F32,
        &device,
    )
    .map_err(E::msg)?;

    let tokenizer = {
        let mut tokenizer = AutoTokenizer::from_local("tmp/tokenizer.json").map_err(E::msg)?;
        // 加入預設的 padding 設定，讓所有的 token 長度對齊一致。
        let params = tokenizers::PaddingParams::default();
        let tokenizer = tokenizer.with_padding(Some(params));
        tokenizer.clone()
    };

    let encoded = tokenizer.encode_batch(vec![
        (
            "How many people live in Berlin?",
            "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",

        ),
        (
            "How many people live in Berlin?",
            "New York City is famous for the Metropolitan Museum of Art.",
        ),
    ], true).map_err(E::msg)?;

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

    // println!("ids: {:?}", ids);
    // println!("type_ids: {:?}", type_ids);
    // println!("attention_mask: {:?}", attention_mask);
    let result = bert.forward(&ids, &type_ids, &attention_mask)?;

    // println!("{:?}", result);
    println!("{:?}", result.to_vec2::<f32>());
    Ok(())
}
