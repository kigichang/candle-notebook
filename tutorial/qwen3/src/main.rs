use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::qwen3::{Config, ModelForCausalLM},
};
use clap::Parser;
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};
use minijinja::{Environment, Template, context};
use serde::Deserialize;
use std::path::PathBuf;
use tokenizers::Tokenizer;

const MODLE_NAME: &str = "Qwen/Qwen3-0.6B";
const MODEL_REVISION: &str = "main";
const MODEL_CACHE_DIR: &str = "hf_cache";

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    // 參考 sample code.
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    println!("candle-ex4: {}:{}", args.model, args.revision);

    // 下載相關檔案
    let repo_files = args.load_model_from_hub()?;

    // 顯示存放的路徑。
    // println!("repo_files:\n{:?}", repo_files);

    let tokenizer =
        tokenizers::Tokenizer::from_file(repo_files.tokenizer).map_err(anyhow::Error::msg)?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&repo_files.model_files, dtype, &device)? };

    let config: Config = serde_json::from_reader(std::fs::File::open(repo_files.config)?)?;
    let model = ModelForCausalLM::new(&config, vb)?;

    // let tokenizer_config: serde_json::Value =
    //     serde_json::from_reader(std::fs::File::open(repo_files.tokenizer_config)?)?;

    // 自 tokenizer_config.json 取得 chat_template
    // let chat_template = tokenizer_config
    //     .get("chat_template")
    //     .map(|v| v.as_str())
    //     .flatten()
    //     .ok_or(anyhow::anyhow!("chat_template not found"))?;
    // if args.show_chat_template {
    //     println!("chat_template:\n{chat_template}");
    // }

    let chat_template = r##"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set content = message.content %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in message.content %}
                {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
                {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- message.content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"##;

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

        generation_config
    };

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        &generation_config,
        template,
        rand::random::<u64>(),
        64,
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

        // 先下載 model.safetensors，如果沒有則找 model.safetensors.index.json
        let model_files = if let Ok(single_file) = repo.get("model.safetensors") {
            vec![single_file]
        } else {
            candle_examples::hub_load_safetensors(&repo, "model.safetensors.index.json")?
        };

        // 下載 generation_config.json
        // 主要取得 eos_token_id
        let generation_config = repo.get("generation_config.json")?;

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
    #[serde(default = "default_repetition_penalty")]
    repetition_penalty: f32,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: usize,
}

fn default_repetition_penalty() -> f32 {
    1.1
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

            enable_thinking => true,

            tokenize => false,
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
