use candle_transformers::generation::{LogitsProcessor, Sampling};

use serde::Deserialize;
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    pub bos_token_id: u32,
    pub pad_token_id: u32,
    pub do_sample: bool,
    pub eos_token_id: Vec<u32>,
    pub repetition_penalty: f32,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
}

impl GenerationConfig {
    pub fn sampling(&self) -> Sampling {
        let temperature = self
            .temperature
            .and_then(|v| if v < 1e-7 { None } else { Some(v) });

        match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match (self.top_k, self.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            },
        }
    }

    pub fn logits_processor(&self, seed: u64) -> LogitsProcessor {
        let sampling = self.sampling();
        LogitsProcessor::from_sampling(seed, sampling)
    }
}
