use clap::Parser;
use mospeada::{Result, repo::Repo};

use candle_transformers::models::quantized_qwen2::ModelWeights;
const MODLE_NAME: &str = "Qwen/Qwen2.5-3B-Instruct";
const MODEL_REVISION: &str = "main";
// const MODEL_CACHE_DIR: &str = "hf_cache";

fn main() -> Result<()> {
    let args = Args::parse();
    let device = mospeada::gpu(0)?;
    mospeada::debug::device_environment(&device);

    let gguf_repo = mospeada::hf_hub::from_pretrained(
        &format!("{}-GGUF", args.model),
        Some(&args.revision),
        None,
        None,
    )?;

    let repo = mospeada::hf_hub::from_pretrained(&args.model, Some(&args.revision), None, None)?;

    let tokenizer = repo.load_tokenizer()?;
    let chat_template = repo.load_chat_template()?;
    let gguf_model_name = MODLE_NAME.split("/").last().unwrap().to_lowercase();
    let model = gguf_repo.load_gguf(
        &format!("{}-q4_0.gguf", gguf_model_name),
        &device,
        ModelWeights::from_gguf,
    )?;

    // 合併 generation 設定
    let generation_config = {
        let mut generation_config = repo.generate_config()?;

        if args.temperature.is_some() {
            generation_config.temperature = args.temperature;
        }

        if args.top_p.is_some() {
            generation_config.top_p = args.top_p;
        }

        if args.top_k.is_some() {
            generation_config.top_k = args.top_k;
        }

        generation_config.repetition_penalty = Some(args.repeat_penalty);

        generation_config
    };

    Ok(())
}

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

    // #[arg(long, default_value_t = String::from(MODEL_CACHE_DIR))]
    // cache_dir: String,
    #[arg(long, default_value_t = String::from("Give me a short introduction to large language model."))]
    prompt: String,

    #[arg(long, default_value_t = false)]
    show_prompt: bool,

    #[arg(long, default_value_t = false)]
    show_chat_template: bool,

    #[arg(long)]
    temperature: Option<f64>,

    #[arg(long)]
    top_p: Option<f64>,

    #[arg(long)]
    top_k: Option<usize>,

    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    #[arg(long, default_value_t = 64)]
    repeat_lan_n: usize,

    #[arg(long, default_value_t = 1024)]
    max_new_tokens: usize,

    #[arg(long)]
    seed: Option<u64>,
}
