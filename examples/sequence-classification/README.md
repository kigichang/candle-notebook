# Sequence Classification

## 1. 模型介紹

[cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) 是給一個問句，從多個句子中找出最相關的句子。通常用在 RAG (Retrieval Augmented Generation) 流程中，Reranking 的步驟。此模型會用到 __BertForSequenceClassification__。但目前 Candle 的 [bert.rs](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/bert.rs) 沒有實作 __BertForSequenceClassification__，所以延續官方的程式碼，自己實作一個。

範例最後的結果，要與模型官方網站的結果一致，我將官方的範例程式放在 [test_seqence_classification.py](test_seqence_classification.py)。

## 2. 實作

### 2.1 官方範例程式與模型檔案

以下是官方的範例程式，用來計算問句與兩個答句的關聯程度。

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('model_name')
tokenizer = AutoTokenizer.from_pretrained('model_name')

features = tokenizer(
    [
        'How many people live in Berlin?', 
        'How many people live in Berlin?'
    ], 
    [
        'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 
        'New York City is famous for the Metropolitan Museum of Art.'
    ],  padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    print(scores)
```

將其中的 `model_name` 換成 `cross-encoder/ms-marco-MiniLM-L-6-v2`，就可以執行這段程式。官方範例程式的答案是：

```python
tensor([[  8.8459],
        [-11.2456]])
```

#### 2.1.1 AutoTokenizer 與 AutoModelForSequenceClassification

在 transformers 套件中，`AutoTokenizer` 與 `AutoModelForSequenceClassification` 會自動對應使用 `BertTokenizerFast` 與 `BertForSequenceClassification`。Python 的 `Tokenizer` 底層用的是 Rust `tokenizers` 套件，因此不需要再實作。如果想知道 Python 是用那個物件，可以用

```python
print(tokenizer)
```

可以得到 tokenizer 物件如下：

```python
BertTokenizerFast(...)
```

#### 2.1.2 官方的模型檔案

可以到 [https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/tree/main](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/tree/main) 查看官方提供的模型檔案。我們會需要：

1. __config.json__: 模型的設定檔。
1. pytorch 或 safetensors 模型檔。
1. __tokenizer.json__: tokenizer 的設定檔。

從官網來看，缺少了 __tokenizer.json__。我們可以透過 Python 的範例程式，來產生 __tokenizer.json__。如下：

```python
tokenizer.save_pretrained("tmp")
```

這樣就會在 `tmp` 資料夾中產生 __tokenizer.json__, __special_tokens_map.json__, __vocab.txt__ 三個檔案。如果模型檔案格式是舊版格式或者有 Shared Tensor 的話，可以參考我的 [bert-base-chinese](https://github.com/kigichang/candle-notebook/blob/main/examples/bert-base-chinese/README.md) 的說明來轉檔解決問題。

### 2.2 模型結構

由於我們要實作 `BertForSequenceClassification`，我們需要先了解它的結構。可以透過以下程式碼來查看。

```python
print(model)
```

得到的結果如下：

```python
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 384, padding_idx=0)
      (position_embeddings): Embedding(512, 384)
      (token_type_embeddings): Embedding(2, 384)
      (LayerNorm): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-5): 6 x BertLayer(
          (attention): BertAttention(
            (self): BertSdpaSelfAttention(
              (query): Linear(in_features=384, out_features=384, bias=True)
              (key): Linear(in_features=384, out_features=384, bias=True)
              (value): Linear(in_features=384, out_features=384, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=384, out_features=384, bias=True)
              (LayerNorm): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=384, out_features=1536, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=1536, out_features=384, bias=True)
            (LayerNorm): LayerNorm((384,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=384, out_features=384, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=384, out_features=1, bias=True)
)
```

如果跟 Candle 的 Bert 程式比對，會發現少了：

1. `BertPooler`:

    ```python
    (pooler): BertPooler(
        (dense): Linear(in_features=384, out_features=384, bias=True)
        (activation): Tanh()
    )
    ```

2. `classifier`:

    ```python
    (classifier): Linear(in_features=384, out_features=1, bias=True)
    ```

因此需要去找 Pyhton 的 [`BertForSequenceClassification`](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1623) 的程式碼，來實作這兩個部分。

為什麼不用理會 `dropout`, Dropout 是訓練的時候會用到，在推論時不會用到，因此我們實作推論時，可以先不理會。

## 3. 實作

實作時，可以直接將官方的檔案複製過來，先修正依賴 crate 錯誤的部分，讓程式可以正常運作，再實作 `BertPooler` 與 `classifier`。

### 3.1 BertPooler

transformers 的 [`BertPooler`](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L737) 程式碼如下：

```python
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

它是一個很基本的神經網路，有一個線性轉換 `dense` 與 激活函數(`activation`) `Tanh`。我們可以用 `candle_nn::linear` 與 `tensor.tanh` 來實作。首先定義模型的結構如下：

```rust
// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L737
pub struct BertPooler {
    dense: Linear,
    activation: fn(&Tensor) -> candle_core::Result<Tensor>,
    span: tracing::Span,
}
```

#### 3.1.1 BertPooler::load

依上面的模型定義，

```python
(pooler): BertPooler(
    (dense): Linear(in_features=384, out_features=384, bias=True)
    (activation): Tanh()
)
```

，與 Python 的程式碼，我們可以實作 `BertPooler::load` 函數如下：

```rust
// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L738
pub fn load(vb: VarBuilder, config: &Config) -> candle_core::Result<Self> {
    let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
    Ok(Self {
        dense,
        activation: |x| x.tanh(),
        span: tracing::span!(tracing::Level::TRACE, "pooler"),
    })
}
```

#### 3.1.2 BertPooler::forward

依 transformers 的程式碼，對應到 Candle 的程式碼如下：

```rust
// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L743
pub fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
    let _enter = self.span.enter();
    let first_token_tensor = hidden_states.i((.., 0))?;
    let pooled_output = self.dense.forward(&first_token_tensor)?;
    let pooled_output = (self.activation)(&pooled_output)?;
    Ok(pooled_output)
}
```

#### 3.1.3 BertPooler 總結

總結程式碼如下：

```rust
// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L737
pub struct BertPooler {
    dense: Linear,
    activation: fn(&Tensor) -> candle_core::Result<Tensor>,
    span: tracing::Span,
}

impl BertPooler {
    // https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L738
    pub fn load(vb: VarBuilder, config: &Config) -> candle_core::Result<Self> {
        let dense = linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        Ok(Self {
            dense,
            activation: |x| x.tanh(),
            span: tracing::span!(tracing::Level::TRACE, "pooler"),
        })
    }

    // https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L743
    pub fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        let _enter = self.span.enter();
        let first_token_tensor = hidden_states.i((.., 0))?.contiguous()?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;
        let pooled_output = (self.activation)(&pooled_output)?;
        Ok(pooled_output)
    }
}
```

### 3.2 修改原本 Candle Bert 程式

接著修改原本 `BertModel` 程式碼，加入 `pooler` 層。由於 `pooler` 非必要，所以我們使用 `Option` 來處理。

```rust
// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>, // 加入 pooler 層
    pub device: Device,
    span: tracing::Span,
}
```

#### 3.2.1 修改 BertModel::load

在 `BertModel::load` 中，加載入 `BertPooler` 的程式碼

```rust
pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
    // 多載入 pooler 層
    let (embeddings, encoder, pooler) = match (
        BertEmbeddings::load(vb.pp("embeddings"), config),
        BertEncoder::load(vb.pp("encoder"), config),
        BertPooler::load(vb.pp("pooler"), config),
    ) {
        (Ok(embeddings), Ok(encoder), pooler) => (embeddings, encoder, pooler),
        (Err(err), _, _) | (_, Err(err), _) => {
            if let Some(model_type) = &config.model_type {
                if let (Ok(embeddings), Ok(encoder), pooler) = (
                    BertEmbeddings::load(vb.pp(format!("{model_type}.embeddings")), config),
                    BertEncoder::load(vb.pp(format!("{model_type}.encoder")), config),
                    BertPooler::load(vb.pp(format!("{model_type}.pooler")), config),
                ) {
                    (embeddings, encoder, pooler)
                } else {
                    return Err(err);
                }
            } else {
                return Err(err);
            }
        }
    };
    Ok(Self {
        embeddings,
        encoder,
        pooler: pooler.ok(),
        device: vb.device().clone(),
        span: tracing::span!(tracing::Level::TRACE, "model"),
    })
}
```

#### 3.2.2 修改 BertModel::forward

判斷是否有 `pooler` 層，如果有就加入 `pooler` 層的推論。

```rust
pub fn forward(
    &self,
    input_ids: &Tensor,
    token_type_ids: &Tensor,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let _enter = self.span.enter();
    let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
    let attention_mask = match attention_mask {
        Some(attention_mask) => attention_mask.clone(),
        None => input_ids.ones_like()?,
    };
    // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L995
    let attention_mask = get_extended_attention_mask(&attention_mask, DType::F32)?;
    let sequence_output = self.encoder.forward(&embedding_output, &attention_mask)?;

    // 加入 pooler 層推論
    // https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1155
    let result = if let Some(ref pooler) = self.pooler {
        pooler.forward(&sequence_output)?
    } else {
        sequence_output
    };
    Ok(result)
}
```

#### 3.2.3 BertModel 總結

```rust
// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L874
pub struct BertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    pooler: Option<BertPooler>, // 加入 pooler 層
    pub device: Device,
    span: tracing::Span,
}

impl BertModel {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // 多載入 pooler 層
        let (embeddings, encoder, pooler) = match (
            BertEmbeddings::load(vb.pp("embeddings"), config),
            BertEncoder::load(vb.pp("encoder"), config),
            BertPooler::load(vb.pp("pooler"), config),
        ) {
            (Ok(embeddings), Ok(encoder), pooler) => (embeddings, encoder, pooler),
            (Err(err), _, _) | (_, Err(err), _) => {
                if let Some(model_type) = &config.model_type {
                    if let (Ok(embeddings), Ok(encoder), pooler) = (
                        BertEmbeddings::load(vb.pp(format!("{model_type}.embeddings")), config),
                        BertEncoder::load(vb.pp(format!("{model_type}.encoder")), config),
                        BertPooler::load(vb.pp(format!("{model_type}.pooler")), config),
                    ) {
                        (embeddings, encoder, pooler)
                    } else {
                        return Err(err);
                    }
                } else {
                    return Err(err);
                }
            }
        };
        Ok(Self {
            embeddings,
            encoder,
            pooler: pooler.ok(),
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "model"),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let embedding_output = self.embeddings.forward(input_ids, token_type_ids)?;
        let attention_mask = match attention_mask {
            Some(attention_mask) => attention_mask.clone(),
            None => input_ids.ones_like()?,
        };
        // https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L995
        let attention_mask = get_extended_attention_mask(&attention_mask, DType::F32)?;
        let sequence_output = self.encoder.forward(&embedding_output, &attention_mask)?;

        // 加入 pooler 層推論
        // https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1155
        let result = if let Some(ref pooler) = self.pooler {
            pooler.forward(&sequence_output)?
        } else {
            sequence_output
        };
        Ok(result)
    }
}
```

### 3.3 實作 BertForSequenceClassification

我們可以先參考 Candle Bert 內的 `BertForMaskedLM` 定義 `BertForSequenceClassification` 模型的結構如下：

```rust
// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1623
pub struct BertForSequenceClassification {
    bert: BertModel,
    classifier: candle_nn::Linear,
}
```

#### 3.3.1 BertForSequenceClassification::load

`BertForSequenceClassification::load` 會載入模型的設定檔，並建立模型的結構。我們可以參考 `BertForMaskedLM::load` 與上面的模型結構，來實作如下：

```rust
// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1624
pub fn load(vb: VarBuilder, config: &Config) -> candle_core::Result<Self> {
    let bert = BertModel::load(vb.pp("bert"), config)?;
    // num_labels 目前沒有支援多個 Label，故固定為 1
    let classifier = candle_nn::linear(config.hidden_size, 1, vb.pp("classifier"))?;
    Ok(Self { bert, classifier })
}
```

1. `bert`: 載入 BertModel。
1. `classifier`: 載入 `Linear`，輸出的維度是 __1__，是因為我們只有一個 Label。

#### 3.3.2 BertForSequenceClassification::forward

在 Bert 層處理完後，接到 `classifier` Python 程式碼如下：

```python
pooled_output = outputs[1]

pooled_output = self.dropout(pooled_output)
logits = self.classifier(pooled_output)
```

由於推論不處理 `dropout`，所以對應到 Candle 的程式碼如下：

```rust
// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1647
pub fn forward(
    &self,
    input_ids: &Tensor,
    token_type_ids: &Tensor,
    attention_mask: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let pooler = self
        .bert
        .forward(input_ids, token_type_ids, attention_mask)?;

    // https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1683
    let logits = self.classifier.forward(&pooler)?;
    Ok(logits)
}
```

#### 3.3.3 BertForSequenceClassification 總結

總結程式碼如下：

```rust
// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1623
pub struct BertForSequenceClassification {
    bert: BertModel,
    classifier: candle_nn::Linear,
}

impl BertForSequenceClassification {
    // https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1624
    pub fn load(vb: VarBuilder, config: &Config) -> candle_core::Result<Self> {
        let bert = BertModel::load(vb.pp("bert"), config)?;
        // num_labels 目前沒有支援多個 Label，故固定為 1
        let classifier = candle_nn::linear(config.hidden_size, 1, vb.pp("classifier"))?;
        Ok(Self { bert, classifier })
    }

    // https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1647
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let pooler = self
            .bert
            .forward(input_ids, token_type_ids, attention_mask)?;

        // https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1683
        let logits = self.classifier.forward(&pooler)?;
        Ok(logits)
    }
}
```

### 3.4 推論

以上的程式準備好後，我們接下來就可以改寫 [test_seqence_classification.py](test_seqence_classification.py) 成 Rust 程式碼。

#### 3.4.1 載入模型

如同先的範例程式，載入模型的程式碼如下：

```rust
let device = candle_notebook::device(false)?;

// 從 HF Hub 下載 Config 和 Model。
let repo = Repo::with_revision(
    "cross-encoder/ms-marco-MiniLM-L-6-v2".to_owned(),
    RepoType::Model,
    "main".to_owned(),
);
let api = Api::new()?;
let api = api.repo(repo);
let config = api.get("config.json")?;
let bert = api.get("pytorch_model.bin")?;

let config = load_config(config)?;
let vb = VarBuilder::from_pth(bert, DType::F32, &device)?;
let bert = BertForSequenceClassification::load(vb, &config)?;
```

#### 3.4.2 載入 Tokenizer

從本地端載入剛剛產生的 `tokenizer.json`。

```rust
// 從本地端讀取匯出的 tokenizer.json。
let mut tokenizer = tokenizers::Tokenizer::from_file("tmp/tokenizer.json").map_err(E::msg)?;
// 加入預設的 padding 設定，讓所有的 token 長度對齊一致。
let params = tokenizers::PaddingParams::default();
let tokenizer = tokenizer.with_padding(Some(params));
```

這邊與先前範例不同，由於比較的答句長度不同，所以我們需要加入 padding 設定，讓答句 token 的長度對齊一致。

#### 3.4.3 輸入問句與答句

這邊需要注意的是，與 Python 輸入方式不同，需要將問句與答句合併成一組 tuple，再使用 encode_batch 方式來編碼。

```rust
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
```

輸入的方式不同，主要是在 Python 中會將問句與答句 `zip` 起來，等同在 Rust 中，將問句與答句放在一個 tuple 中。這一段的處理程式在 [PreTrainedTokenizerBase._call_one](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/tokenization_utils_base.py#L3036)  如下：

```python
# https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/tokenization_utils_base.py#L3108
batch_text_or_text_pairs = list(zip(text, text_pair)) if text_pair is not None else text
return self.batch_encode_plus(
            batch_text_or_text_pairs=batch_text_or_text_pairs,
            ...
)
```

其中 `text` 是問句 list，`text_pair` 是答句 list。

#### 3.3.4 推論

取得 token 並進行推論。

```rust
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

let result = bert.forward(&ids, &type_ids, Some(&attention_mask))?;
println!("{:?}", result.to_vec2::<f32>());
```

#### 3.3.5 執行程式

執行

```shell
$ cargo run --release --example sequence-classification
Ok([[8.845854], [-11.24556]])
```

與官方的結果一致。

## 4. 復盤

1. 如何產生 __tokenizer.json__。
1. 如何解析預訓練模型結構。
1. 如何參考 Python transformers 程式碼，實作 Rust 程式碼。
1. 如何對齊句子的 token 長度。
1. 如何使用 __BertForSequenceClassification__。
