use anyhow::Result;
use candle_core::{IndexOp, Tensor};
use macross::models::auto::{AutoModel, AutoTokenizer};

struct XLMRobertaModel;

impl AutoModel for XLMRobertaModel {
    type Config = candle_transformers::models::xlm_roberta::Config;
    type Model = candle_transformers::models::xlm_roberta::XLMRobertaModel;

    fn load(vb: candle_nn::VarBuilder, config: &Self::Config) -> candle_core::Result<Self::Model>
    where
        Self::Model: Sized,
    {
        candle_transformers::models::xlm_roberta::XLMRobertaModel::new(config, vb)
    }
}

fn main() -> Result<()> {
    const MODLE_NAME: &str = "BAAI/bge-m3";

    let device = macross::device(false)?;

    let tokenizer = {
        let mut tokenizer =
            AutoTokenizer::from_pretrained(MODLE_NAME).map_err(anyhow::Error::msg)?;
        let params = tokenizers::PaddingParams::default();
        //println!("padding: {:?}", params);
        let truncation = tokenizers::TruncationParams::default();
        //println!("truncate: {:?}", truncation);
        let tokenizer = tokenizer.with_padding(Some(params));
        let tokenizer = tokenizer
            .with_truncation(Some(truncation))
            .map_err(anyhow::Error::msg)?;
        tokenizer.clone()
    };
    // println!("tokenizer: {:?}", tokenizer);

    let bert =
        XLMRobertaModel::from_pretrained((MODLE_NAME, true), candle_core::DType::F32, &device)?;

    let s1 = {
        let sentences = vec!["What is BGE M3?", "Defination of BM25"];
        let encoded_input = tokenizer
            .encode_batch(sentences, true)
            .map_err(anyhow::Error::msg)?;

        let ids = encoded_input
            .iter()
            .map(|e| Tensor::new(e.get_ids(), &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let ids = Tensor::stack(&ids, 0)?;
        // println!("ids: {:?}", ids.to_vec2::<u32>()?);

        let type_ids = encoded_input
            .iter()
            .map(|e| Tensor::new(e.get_type_ids(), &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let type_ids = Tensor::stack(&type_ids, 0)?;
        // println!("type_ids: {:?}", type_ids.to_vec2::<u32>()?);

        let attention_mask = encoded_input
            .iter()
            .map(|e| Tensor::new(e.get_attention_mask(), &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        //println!("attention_mask: {:?}", attention_mask.to_vec2::<u32>()?);

        //let result = bert.forward(&ids, &type_ids, &attention_mask)?;
        let result = bert.forward(&ids, &attention_mask, &type_ids, None, None, None)?;
        let result = result.i((.., 0))?.contiguous()?;
        let result = macross::normalize(&result)?;
        macross::print_tensor::<f32>(&result)?;
        result
    };

    let s2 = {
        let sentences = vec!["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"];
        let encoded_input = tokenizer
            .encode_batch(sentences, true)
            .map_err(anyhow::Error::msg)?;

        let ids = encoded_input
            .iter()
            .map(|e| Tensor::new(e.get_ids(), &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let ids = Tensor::stack(&ids, 0)?;
        // println!("ids: {:?}", ids.to_vec2::<u32>()?);

        let type_ids = encoded_input
            .iter()
            .map(|e| Tensor::new(e.get_type_ids(), &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let type_ids = Tensor::stack(&type_ids, 0)?;
        // println!("type_ids: {:?}", type_ids.to_vec2::<u32>()?);

        let attention_mask = encoded_input
            .iter()
            .map(|e| Tensor::new(e.get_attention_mask(), &device))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        //println!("attention_mask: {:?}", attention_mask.to_vec2::<u32>()?);

        //let result = bert.forward(&ids, &type_ids, &attention_mask)?;
        let result = bert.forward(&ids, &attention_mask, &type_ids, None, None, None)?;
        let result = result.i((.., 0))?.contiguous()?;
        let result = macross::normalize(&result)?;
        macross::print_tensor::<f32>(&result)?;
        result
    };

    let result = s1.matmul(&s2.t()?)?;
    macross::print_tensor::<f32>(&result)?;
    //println!("result: {:?}", result.i(1)?.to_vec1::<f32>()?);
    //println!("result: {:?}", result.to_vec2::<f32>()?);
    //let result = result.mean(1)?;
    // let mean = result.mean(1)?;
    // let result = macross::normalize(&mean)?;
    //println!("result: {:?}", result.to_vec2::<f32>()?);
    Ok(())
}
