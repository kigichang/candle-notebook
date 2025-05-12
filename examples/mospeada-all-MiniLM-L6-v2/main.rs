use candle_core::Tensor;
use mospeada::{repo::Repo, Result};
use tokenizers;

fn main() -> Result<()> {
    println!("mospeada-all-MiniLM-L6-v2");

    let device = mospeada::gpu(0)?;
    let sentences = vec!["This is an example sentence", "Each sentence is converted"];

    let repo = mospeada::hf_hub::from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        None,
        None,
        None,
    )?;

    let tokenizer = {
        let mut tokenizer = repo.load_tokenizer()?;

        let padding = tokenizers::PaddingParams::default();
        tokenizer.with_padding(Some(padding));

        let truncation = tokenizers::TruncationParams::default();
        tokenizer.with_truncation(Some(truncation))?;

        println!("tokenizer: {:?}", tokenizer.get_padding());
        println!("tokenizer: {:?}", tokenizer.get_truncation());
        tokenizer
    };

    let bert = repo.load_model(candle_core::DType::F32, &device, |config, vb| {
        candle_transformers::models::bert::BertModel::load(vb, config)
    })?;

    let encoded_input = tokenizer.encode_batch(sentences, true)?;

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
    println!("result: {:?}", result.shape());
    // mospeada::debug::print_tensor::<f32>(&result)?;
    let mean = mospeada::pooling::mean(&result, &attention_mask)?;
    let result = mospeada::normalize(&mean)?;
    //println!("result: {:?}", result.to_vec2::<f32>()?);
    mospeada::debug::print_tensor::<f32>(&result)?;

    Ok(())
}
