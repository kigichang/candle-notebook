use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::qwen2::{Config, ModelForCausalLM},
};
use clap::Parser;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use minijinja::{Environment, Template, context};
use serde::Deserialize;
use std::path::PathBuf;
use tokenizers::Tokenizer;

const MODLE_NAME: &str = "Qwen/Qwen2.5-1.5B-Instruct";
const MODEL_REVISION: &str = "main";
const MODEL_CACHE_DIR: &str = "hf_cache";

#[derive(Debug, Parser)]
struct Args {
    #[arg(long, default_value_t = String::from("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."))]
    system: String,

    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = String::from(MODLE_NAME))]
    model: String,

    #[arg(long, default_value_t = String::from(MODEL_REVISION))]
    revision: String,

    #[arg(long, default_value_t = String::from(MODEL_CACHE_DIR))]
    cache_dir: String,

    #[arg(long, default_value_t = String::from("Give me a short introduction to large language model."))]
    prompt: String,

    #[arg(long, default_value_t = false)]
    show_prompt: bool,
}

impl Args {
    fn load_model_from_hub(&self) -> Result<RepoFiles> {
        let api = ApiBuilder::new()
            .with_cache_dir((&self.cache_dir).into())
            .build()?;

        let repo = Repo::with_revision(
            self.model.to_owned(),
            RepoType::Model,
            self.revision.to_owned(),
        );

        let repo = api.repo(repo);

        let tokenizer_config = repo.get("tokenizer_config.json")?; // 下載 tokenizer_config.json
        let tokenizer = repo.get("tokenizer.json")?; // 下載 tokenizer.json
        let config = repo.get("config.json")?; // 下載 config.json

        let model_files = if let Ok(single_file) = repo.get("model.safetensors") {
            vec![single_file]
        } else {
            candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
        };
        let generation_config = repo.get("generation_config.json")?; // 下載 generation_config.json

        Ok(RepoFiles {
            tokenizer_config,
            tokenizer,
            config,
            model_files,
            generation_config,
        })
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    println!("candle-ex4: {}:{}", args.model, args.revision);

    let repo_files = args.load_model_from_hub()?;

    // 顯示存放的路徑。
    // println!("repo_files:\n{:?}", repo_files);

    let tokenizer_config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(repo_files.tokenizer_config)?)?;

    let chat_template = tokenizer_config
        .get("chat_template")
        .map(|v| v.as_str())
        .flatten()
        .ok_or(anyhow::anyhow!("chat_template not found"))?;

    // println!("chat_template: {chat_template}");

    let mut env = Environment::new();
    env.add_template("chat_template", chat_template)?;
    let template = env.get_template("chat_template")?;

    let tokenizer =
        tokenizers::Tokenizer::from_file(repo_files.tokenizer).map_err(anyhow::Error::msg)?;
    let config: Config = serde_json::from_reader(std::fs::File::open(repo_files.config)?)?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&repo_files.model_files, dtype, &device)? };

    let model = ModelForCausalLM::new(&config, vb)?;

    let generation_config: GenerationConfig =
        serde_json::from_reader(std::fs::File::open(repo_files.generation_config)?)?;

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        &generation_config,
        template,
        299792458,
        64,
        &device,
    );

    let sample_len = 10240;

    pipeline.run(&args.system, &args.prompt, sample_len, args.show_prompt)?;

    Ok(())
}

#[derive(Debug)]
struct RepoFiles {
    tokenizer_config: PathBuf,
    tokenizer: PathBuf,
    config: PathBuf,
    model_files: Vec<PathBuf>,
    generation_config: PathBuf,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct GenerationConfig {
    bos_token_id: u32,
    pad_token_id: u32,
    do_sample: bool,
    eos_token_id: Vec<u32>,
    repetition_penalty: f32,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: usize,
}

// 修改自 https://github.com/huggingface/candle/blob/main/candle-examples/examples/qwen/main.rs#L34
struct TextGeneration<'template> {
    model: ModelForCausalLM,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token_id: Vec<u32>,
    template: Template<'template, 'template>,
}

// 修改自 https://github.com/huggingface/candle/blob/main/candle-examples/examples/qwen/main.rs#L43
impl<'template> TextGeneration<'template> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelForCausalLM,
        tokenizer: Tokenizer,
        config: &GenerationConfig,
        template: Template<'template, 'template>,
        seed: u64,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor =
            LogitsProcessor::new(seed, config.temperature.clone(), config.top_p.clone());

        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            template,
            logits_processor,
            repeat_penalty: config.repetition_penalty,
            repeat_last_n,
            device: device.clone(),
            eos_token_id: config.eos_token_id.clone(),
        }
    }

    fn run(
        &mut self,
        system: &str,
        prompt: &str,
        sample_len: usize,
        show_prompt: bool,
    ) -> Result<()> {
        use std::io::Write;
        let prompt = self.template.render(context! {
            messages => vec![
                context!{
                    role => "system",
                    content => system,
                },
                context! {
                    role => "user",
                    content => prompt,
                }
            ],
            add_generation_prompt => true,
        })?;

        if show_prompt {
            println!("prompt:\n{prompt}");
        }

        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        // for &t in tokens.iter() {
        //     if let Some(t) = self.tokenizer.next_token(t)? {
        //         print!("{t}")
        //     }
        // }
        // std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if self.eos_token_id.contains(&next_token) {
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}
