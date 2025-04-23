  
# Generation with GGUF
  
本章節主要介紹如何使用 Huggingface Candle 與 [Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF ) 進行文字生成。本範例是修改自 [https://github.com/huggingface/candle/tree/main/candle-examples/examples/quantized-qwen2-instruct](https://github.com/huggingface/candle/tree/main/candle-examples/examples/quantized-qwen2-instruct )。在使用預訓練模型時，可以先到 [https://github.com/huggingface/candle/tree/main/candle-examples/examples](https://github.com/huggingface/candle/tree/main/candle-examples/examples ) 的尋找有沒有範例可以參考。
  
## 程式碼流程與說明
  
### 1. 自 Huggingface 下載相關檔案
  
使用的 `tokenizer_config.json`，`tokenizer.json`，與`generation_config.json` 需要從 [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct ) 下載取得。
  
### 2. 載入 Tokenizer 與 Chat Template
  
與 [4.generation-llama](../4.generation-llama/README.md ) 範例類似。
  
### 3. 下載 GGUF 檔案並載入模型
  
本範例使用的模型是 __qwen2.5-3b-instruct-q4_0.gguf__，下載自 [Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF )。依範例中的程式碼，載入模型：
  
```rust
let model = {
    let model =
        gguf_file::Content::read(&mut model_file).map_err(|e| e.with_path(model_path))?;
    let mut total_size_in_bytes = 0;
    for (_, tensor) in model.tensor_infos.iter() {
        let elem_count = tensor.shape.elem_count();
        total_size_in_bytes +=
            elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
    }
    println!(
        "loaded {:?} tensors ({}) in {:.2}s",
        model.tensor_infos.len(),
        &format_size(total_size_in_bytes),
        start.elapsed().as_secs_f32(),
    );
  
    ModelWeights::from_gguf(model, &mut model_file, &device)?
};
```
  
1. 使用 `candle_core::quantized::gguf_file::Content` 來載入 GGUF 檔案。
1. 使用 `candle_transformers::models::quantized_qwen2::ModelWeights` 載入模型權重。
  
### 4. 生成文字
  
與 [4.generation-llama](../4.generation-llama/README.md ) 範例類似。
  
## 完整的程式碼
  
```rust
use anyhow::Result;
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_examples::token_output_stream::TokenOutputStream;
  
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_qwen2::ModelWeights;
use clap::Parser;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use minijinja::{Environment, Template, context};
use serde::Deserialize;
use std::path::PathBuf;
use tokenizers::Tokenizer;
  
const MODLE_NAME: &str = "Qwen/Qwen2.5-3B-Instruct";
const MODEL_REVISION: &str = "main";
const MODEL_CACHE_DIR: &str = "hf_cache";
  
fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
  
    println!("candle-ex5: {}-GGUF:{}", args.model, args.revision);
    candle_notebook::device_environment(&device);
  
    // 下載相關檔案
    let repo_files = args.load_model_from_hub()?;
  
    // 顯示存放的路徑。
    // println!("repo_files:\n{:?}", repo_files);
  
    let tokenizer =
        tokenizers::Tokenizer::from_file(repo_files.tokenizer).map_err(anyhow::Error::msg)?;
  
    // 自 tokenizer_config.json 取得 chat_template
    let tokenizer_config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(repo_files.tokenizer_config)?)?;
    let chat_template = tokenizer_config
        .get("chat_template")
        .map(|v| v.as_str())
        .flatten()
        .ok_or(anyhow::anyhow!("chat_template not found"))?;
    if args.show_chat_template {
        println!("chat_template:\n{chat_template}");
    }
  
    // 載入 GGUF 模型檔案
    let model_path = &repo_files.model_files[0];
    let mut model_file = std::fs::File::open(model_path)?;
    let start = std::time::Instant::now();
    let model = {
        let model =
            gguf_file::Content::read(&mut model_file).map_err(|e| e.with_path(model_path))?;
        let mut total_size_in_bytes = 0;
        for (_, tensor) in model.tensor_infos.iter() {
            let elem_count = tensor.shape.elem_count();
            total_size_in_bytes +=
                elem_count * tensor.ggml_dtype.type_size() / tensor.ggml_dtype.block_size();
        }
        println!(
            "loaded {:?} tensors ({}) in {:.2}s",
            model.tensor_infos.len(),
            &format_size(total_size_in_bytes),
            start.elapsed().as_secs_f32(),
        );
  
        ModelWeights::from_gguf(model, &mut model_file, &device)?
    };
  
    // 使用 minijinja 來處理 chat_template
    // minijinja::Environment 類似 Global Template Engine,
    // 可以加入多個子 template。
    let mut env = Environment::new();
    env.add_template("qwen_chat_template", chat_template)?;
  
    // 取得 Qwen 的 template
    let template = env.get_template("qwen_chat_template")?;
  
    // 合併 generation 設定
    let generation_config = {
        let mut generation_config: GenerationConfig =
            serde_json::from_reader(std::fs::File::open(repo_files.generation_config)?)?;
  
        if args.temperature.is_some() {
            generation_config.temperature = args.temperature;
        }
  
        if args.top_p.is_some() {
            generation_config.top_p = args.top_p;
        }
  
        if args.top_k.is_some() {
            generation_config.top_k = args.top_k;
        }
  
        generation_config.repetition_penalty = args.repeat_penalty;
  
        generation_config
    };
  
    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        &generation_config,
        template,
        rand::random::<u64>(),
        args.repeat_lan_n,
        &device,
    );
  
    pipeline.run(
        &args.system,
        &args.prompt,
        args.max_new_tokens,
        args.show_prompt,
    )?;
  
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
  
    #[arg(long, default_value_t = String::from(MODEL_CACHE_DIR))]
    cache_dir: String,
  
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
}
  
impl Args {
    /// 下載模型檔案
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
  
        // 下載 tokenizer_config.json
        // 主要要取得 chat_template
        let tokenizer_config = repo.get("tokenizer_config.json")?;
        let tokenizer = repo.get("tokenizer.json")?; // 下載 tokenizer.json
        let config = repo.get("config.json")?; // 下載 config.json
  
        // 下載 generation_config.json
        // 主要取得 eos_token_id
        let generation_config = repo.get("generation_config.json")?;
  
        let repo = Repo::with_revision(
            format!("{}-GGUF", self.model),
            RepoType::Model,
            self.revision.to_owned(),
        );
  
        let repo = api.repo(repo);
        let model_name = self.model.split('/').last().unwrap().to_lowercase();
        let model_name = format!("{}-q4_0.gguf", model_name);
        println!("download: {model_name}");
        let model_files = vec![repo.get(&model_name)?];
  
        Ok(RepoFiles {
            tokenizer_config,
            tokenizer,
            config,
            model_files,
            generation_config,
        })
    }
}
  
/// HF hub repo 檔案
#[allow(dead_code)]
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
    top_k: Option<usize>,
}
  
impl GenerationConfig {
    fn sampling(&self) -> Sampling {
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
  
    fn logits_processor(&self, seed: u64) -> LogitsProcessor {
        let sampling = self.sampling();
        LogitsProcessor::from_sampling(seed, sampling)
    }
}
  
// 修改自 https://github.com/huggingface/candle/blob/main/candle-examples/examples/qwen/main.rs#L34
struct TextGeneration<'template> {
    model: ModelWeights,
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
        model: ModelWeights,
        tokenizer: Tokenizer,
        config: &GenerationConfig,
        template: Template<'template, 'template>,
        seed: u64,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = config.logits_processor(seed);
  
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
        max_new_tokens: usize,
        show_prompt: bool,
    ) -> Result<()> {
        use std::io::Write;
        // 等同官網 sample code
        // prompt = "Give me a short introduction to large language model."
        // messages = [
        //     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        //     {"role": "user", "content": prompt}
        // ]
        // text = tokenizer.apply_chat_template(
        //     messages,
        //     tokenize=False,
        //     add_generation_prompt=True
        // )
        let text = self.template.render(context! {
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
            // 依 chat_template 的定義，加入 add_generation_prompt
            // 會在最後加 '<|im_start|>assistant\n'
            // 才會正確生成答案，否則在生成時，會出現奇怪的 token，eg: 'ystem\n'
            add_generation_prompt => true,
        })?;
  
        if show_prompt {
            println!("prompt:\n{text}");
        }
  
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(text, true)
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
        for index in 0..max_new_tokens {
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
  
fn format_size(size_in_bytes: usize) -> String {
    if size_in_bytes < 1_000 {
        format!("{}B", size_in_bytes)
    } else if size_in_bytes < 1_000_000 {
        format!("{:.2}KB", size_in_bytes as f64 / 1e3)
    } else if size_in_bytes < 1_000_000_000 {
        format!("{:.2}MB", size_in_bytes as f64 / 1e6)
    } else {
        format!("{:.2}GB", size_in_bytes as f64 / 1e9)
    }
}
  
```  
  