use anyhow::{Error as E, Result};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::{
    bert::{BertModel, Config},
    with_tracing::{linear, Linear},
};
use std::fs::File;
use std::path::Path;
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let device = candle_notebook::device(false)?;

    let (tokenizer_filename, config_filename, model_filename) = candle_notebook::load_from_hf_hub(
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "main",
        "vocab.txt",
        "config.json",
        "pytorch_model.bin",
    )?;

    let config = load_config(config_filename)?;
    let vb = VarBuilder::from_pth(model_filename, DType::F32, &device)?;
    let bert = BertForSequenceClassification::load(vb, &config, 384, 1)?;

    let mut tokenizer = tokenizers::Tokenizer::from_file("tmp/tokenizer.json").map_err(E::msg)?;
    // let mut tokenizer =
    //     Tokenizer::from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2", None).map_err(E::msg)?;
    let params = tokenizers::PaddingParams::default();
    let tokenizer = tokenizer.with_padding(Some(params));

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

    // let ids = Tensor::new(
    //     &[
    //         [
    //             101u32, 2129, 2116, 2111, 2444, 1999, 4068, 1029, 102, 4068, 2038, 1037, 2313,
    //             1997, 1017, 1010, 19611, 1010, 6021, 2487, 5068, 4864, 1999, 2019, 2181, 1997,
    //             6486, 2487, 1012, 6445, 2675, 7338, 1012, 102,
    //         ],
    //         [
    //             101, 2129, 2116, 2111, 2444, 1999, 4068, 1029, 102, 2047, 2259, 2103, 2003, 3297,
    //             2005, 1996, 4956, 2688, 1997, 2396, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //         ],
    //     ],
    //     &device,
    // )?;

    // let type_ids = Tensor::new(
    //     &[
    //         [
    //             0u32, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    //             1, 1, 1, 1, 1, 1, 1,
    //         ],
    //         [
    //             0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    //             0, 0, 0, 0, 0, 0,
    //         ],
    //     ],
    //     &device,
    // )?;

    // let attention_mask = Tensor::new(
    //     &[
    //         [
    //             1u32, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    //             1, 1, 1, 1, 1, 1, 1,
    //         ],
    //         [
    //             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
    //             0, 0, 0, 0, 0, 0,
    //         ],
    //     ],
    //     &device,
    // )?;

    println!("ids: {:?}", ids);
    println!("type_ids: {:?}", type_ids);
    println!("attention_mask: {:?}", attention_mask);
    let result = bert.forward(&ids, &type_ids, Some(&attention_mask))?;

    println!("{:?}", result);
    println!("{:?}", result.to_vec2::<f32>());
    //println!("{:?}", result.to_vec3::<f32>());
    Ok(())
}

// class BertPooler(nn.Module):
//     def __init__(self, config):
//         super().__init__()
//         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
//         self.activation = nn.Tanh()

//     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
//         # We "pool" the model by simply taking the hidden state corresponding
//         # to the first token.
//         first_token_tensor = hidden_states[:, 0]
//         pooled_output = self.dense(first_token_tensor)
//         pooled_output = self.activation(pooled_output)
//         return pooled_output

pub struct BertPooler {
    dense: Linear,
    activation: fn(&Tensor) -> candle_core::Result<Tensor>,
}

impl BertPooler {
    pub fn load(vb: VarBuilder, _config: &Config, hidden_size: usize) -> candle_core::Result<Self> {
        let dense = linear(hidden_size, hidden_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            activation: |x| x.tanh(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let first_token_tensor = hidden_states.i((.., 0))?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;
        let pooled_output = (self.activation)(&pooled_output)?;
        Ok(pooled_output)
    }
}

pub struct BertForSequenceClassification {
    bert: BertModel,
    pooler: BertPooler,
    classifier: candle_nn::Linear,
}

impl BertForSequenceClassification {
    pub fn load(
        vb: VarBuilder,
        config: &Config,
        hidden_size: usize,
        num_lables: usize,
    ) -> candle_core::Result<Self> {
        let bert = BertModel::load(vb.pp("bert"), config)?;
        let pooler = BertPooler::load(vb.pp("bert").pp("pooler"), config, hidden_size)?;
        let classifier = candle_nn::linear(hidden_size, num_lables, vb.pp("classifier"))?;
        Ok(Self {
            bert,
            pooler,
            classifier,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let sequence_output = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;
        println!("sequence_output: {:?}", sequence_output);
        let pooler = self.pooler.forward(&sequence_output)?;
        println!("pooler: {:?}", pooler);
        let logits = self.classifier.forward(&pooler)?;
        println!("logits: {:?}", logits);
        Ok(logits)
    }
}

fn load_config<P: AsRef<Path>>(config_filename: P) -> Result<Config> {
    let config = File::open(config_filename)?;
    Ok(serde_json::from_reader(config)?)
}
