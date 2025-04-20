---
export_on_save:
  markdown: true
markdown:
  image_dir: assets
  path: README.md
  ignore_from_front_matter: true
  absolute_image_path: false
---

# Generation

本章節主要介紹如何使用 Huggingface Candle 與 [Qwen2.5-1.5-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 進行文字生成。本範例是修改自 [https://github.com/huggingface/candle/tree/main/candle-examples/examples/qwen](https://github.com/huggingface/candle/tree/main/candle-examples/examples/qwen)。在使用預訓練模型時，可以先到 [https://github.com/huggingface/candle/tree/main/candle-examples/examples](https://github.com/huggingface/candle/tree/main/candle-examples/examples) 的尋找有沒有範例可以參考。

## 程式碼流程與說明

### 1. 自 Huggingface 下載相關檔案

以先前範例雷同，但

1. 多下載了 `tokenizer_config.json`，讀取模型的 chat template。
1. 多下載了 `generation_config.json`，讀取 `eos_token_id`，用來判斷結束生成條件。
1. 先下載 `model.safetensors`，如果沒有，代表該模型檔案太大，有切割成多個檔案，這時候就要下載 `model.safetensors.index.json`，然後再下載切割的模型檔案。
    - 使用 `candle_examples::hub_load_safetensors` 來處理最後存放的路徑。

### 2. 載入 Tokenizer 與預訓練模型

與先前範例雷同。

1. 使用 `tokenizers` 載入 `tokenizer.json`，產生 `Tokenizer`。
1. 使用 `VarBuilder::from_mmaped_safetensors` 載入模型檔案。仿照官方 Sample Code。用 Device 是否為 CUDA 來決定是否使用 `BF16`。

    ```rust
    let device = candle_examples::device(args.cpu)?;
    // 參考 sample code.
    let dtype = if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    };

    let tokenizer =
        tokenizers::Tokenizer::from_file(repo_files.tokenizer).map_err(anyhow::Error::msg)?;
    let vb =
        unsafe { VarBuilder::from_mmaped_safetensors(&repo_files.model_files, dtype, &device)? };
    ```

1. 本次範例使用 __Qwen2.5-1.5-Instruct__ 會用到
    - `candle_transformers::models::qwen2::Config`: 載入 [https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) 預訓練模型的設定檔。
    - `candle_transformers::models::qwen2::ModelForCausalLM`: 自 `VarBuilder` 來載入預訓練模型。

1. 怎麼決定使用何種預訓練模型結構呢？可以參考 [https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/config.json) 的 `architectures` 參數。

    ```json
    "architectures": [
      "Qwen2ForCausalLM"
    ],
    ```

    比如 __Qwen2.5-1.5B-Instruct__ 是 `Qwen2ForCausalLM`，所以使用 `candle_transformers::models::qwen2::ModelForCausalLM` 來載入預訓練模型。

    ```rust
    let config: Config = serde_json::from_reader(std::fs::File::open(repo_files.config)?)?;
    let model = ModelForCausalLM::new(&config, vb)?;
    ```

### 3. 使用 Chat Template

在 [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) 的 Model Card，有一段文字生成前的 Chat Template 範例：

```python
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

可以透過 `tokenizer_config.json` 來取得 `chat_template`，如下：

```shell
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}
    {%- else %}
        {{- '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\n<tool_call>\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
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
{%- endif %}
```

可以使用 [MiniJinja](https://github.com/mitsuhiko/minijinja) 當做 Template Engine，來輸出生成前的文字。流程如下：

```rust
// 自 tokenizer_config.json 取得 chat_template
let chat_template = tokenizer_config
    .get("chat_template")
    .map(|v| v.as_str())
    .flatten()
    .ok_or(anyhow::anyhow!("chat_template not found"))?;

// 使用 minijinja 來處理 chat_template
// minijinja::Environment 類似 Global Template Engine,
// 可以加入多個子 template。
let mut env = Environment::new();
env.add_template("qwen_chat_template", chat_template)?;

// 取得 Qwen 的 template
let template = env.get_template("qwen_chat_template")?;
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
```

需要注意的是 __add_generation_prompt__ 要設成 __true__，則輸出的 text 會在最後加上 `<|im_start|>assistant\n`，否則在生成文字時會出現 `ystem\n`。

## 4. 生成文字

生成文字的流程，修改 [https://github.com/huggingface/candle/blob/main/candle-examples/examples/qwen/main.rs](https://github.com/huggingface/candle/blob/main/candle-examples/examples/qwen/main.rs) 的 `TextGeneration`。

1. 新加 chat template。
1. 使用 GenerationConfig 取代原本的 `temperature` 與 `top_p` 輸入，以及取得 __eos_token_id__ 的設定。
1. __修改結束生成的條件。__

原本結束生成的條件是判斷是否取得 __<|endoftext|>__，如下：

```rust
let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
    Some(token) => token,
    None => anyhow::bail!("cannot find the <|endoftext|> token"),
};

...

let next_token = self.logits_processor.sample(&logits)?;
tokens.push(next_token);
generated_tokens += 1;
if next_token == eos_token {
    break;
}
```

但有時候會發生一直重覆生成前面的答案。由 __generation_config.json__ 的設定檔發現 `eos_token_id` 有兩個 __<|im_end|>__ (151645) 與  __<|endoftext|>__ (151643)：

```json
eos_token_id": [
  151645,
  151643
],
```

所以修改判斷結束生成的條件如下：

```rust
if self.eos_token_id.contains(&next_token) {
    break;
}
```

避免一直重覆生成前面的答案。

## 重點整理

1. 看 config.json 的 `architectures` 來決定使用何種預訓練模型結構。
1. [MiniJinja](https://github.com/mitsuhiko/minijinja) 當做 Template Engine，來輸出生成前的文字。
1. 留意 __add_generation_prompt__ 要設成 __true__，則輸出的 text 會在最後加上 `<|im_start|>assistant\n`，否則在生成文字時會出現 `ystem\n`。
1. 留意 __generation_config.json__ 的 `eos_token_id` 有那些，調整判斷結束生成的條件。

## 完整程式碼

@import "src/main.rs" {as=rust}
