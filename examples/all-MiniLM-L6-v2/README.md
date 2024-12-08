# Sentence Transforms - All-MiniLM-L6-v2

Text Embedding 是將句子或文章對應到指定長度的向量空間上，用來計算句子或文章之間的相似度，也是 RAG 的第一個步驟。這個範例我們實作 [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 來實現 Text Embedding。

Huggingface 的 [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference)，提供常用的 Text Embedding 模型，可以自行架設服務。這個範例也有參考裏面的實作，需注意的是 text-embeddings-inference 用的不是官方版本的 Candle。

## 1. 模型介紹

[sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 會將句子轉換成 __384__ 維的向量。

## 2. 官方範例

以下是官方範例，`AutoTokenizer` 與 `AutoModel` 的說明，請見 [cross-encoder](../cross-encoder/README.md)。最後要產出與官方範例一樣的結果。

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# print(tokenizer)
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# print(model)

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# print("encoded_input:")
# print(encoded_input)

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)

# print("model_output[0]:")
# print(model_output[0])
# print("model_output[1]:")
# print(model_output[0])

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(sentence_embeddings)
```

### 2.1 說明

__sentence-transformers/all-MiniLM-L6-v2__ 是 Bert 模型。我們可以透過 `BertModel` 來進行推論，最後再透過 `mean_pooling` 與 `F.normalize` 來得到句子的向量。

## 3. 實作

為方便之後 Candle 操作，我實作一個套件 [macross](https://github.com/kigichang/macross)，它是基於 [Huggingface hub](https://github.com/huggingface/hf-hub)，實作有關下載、載入模型與模型推論的功能。

### 3.1 下載模型

仿造 transformers 的 `AutoModel` 與 `AutoTokenizer`，我也實作類似的功能下載與載入 tokenizer 與 model。你也可以仿照先前的範例，自行下載與載入模型。

### 3.2 修改 [cross-encoder/bert.rs](../cross-encoder/bert.rs) 程式

我將 [cross-encoder/bert.rs](../cross-encoder/bert.rs) 複製到 [macross/src/models/bert.rs](https://github.com/kigichang/macross/blob/main/src/models/bert.rs)，進行修改。

#### 3.2.1 修改 Config

`Config` 新加 `id2label: Option<HashMap<String, String>>`。

```rust
// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/configuration_bert.py#L99
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    vocab_size: usize,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    pub hidden_act: HiddenAct,
    hidden_dropout_prob: f64,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    initializer_range: f64,
    layer_norm_eps: f64,
    pad_token_id: usize,
    #[serde(default)]
    position_embedding_type: PositionEmbeddingType,
    #[serde(default)]
    use_cache: bool,
    classifier_dropout: Option<f64>,
    id2label: Option<HashMap<String, String>>,
    model_type: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: true,
            classifier_dropout: None,
            id2label: None,
            model_type: Some("bert".to_string()),
        }
    }
}
```

#### 3.2.2 修改 BertModel 的 forward

修改 `BertModel` 的 `forward` 函數，雷同 transformers 的 `BertModel` 的 `forward` 函數，回傳 `sequence_output` 與 `pooled_output`。

```rust
pub fn forward(
    &self,
    input_ids: &Tensor,
    token_type_ids: &Tensor,
    attention_mask: Option<&Tensor>,
) -> Result<(Tensor, Option<Tensor>)> {
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
    let pooled_ouput = if let Some(pooler) = &self.pooler {
        Some(pooler.forward(&sequence_output)?)
    } else {
        None
    };
    Ok((sequence_output, pooled_ouput))
}
```

#### 3.2.3 修正 BertForMaskedLM 的 forward

修改 `BertForMaskedLM` 的 `forward` 函數，使用 `sequence_output` 來進行推論。

```rust
pub fn forward(
    &self,
    input_ids: &Tensor,
    token_type_ids: &Tensor,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let sequence_output = self
        .bert
        .forward(input_ids, token_type_ids, attention_mask)?;
    self.cls.forward(&sequence_output.0)
}
```

#### 3.2.4 修改 BertForSequenceClassification

修改 `BertForSequenceClassification` 的 `load` 加入 `id2label` 取的 label 的數量。

```rust
pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
    let num_labels = if let Some(id2label) = &config.id2label {
        id2label.len()
    } else {
        candle_core::bail!("id2label is required for BertForSequenceClassification")
    };

    let bert = BertModel::load(vb.pp("bert"), config)?;
    let dropout = Dropout::new(if let Some(pr) = config.classifier_dropout {
        pr
    } else {
        config.hidden_dropout_prob
    });

    let classifier = candle_nn::linear(config.hidden_size, num_labels, vb.pp("classifier"))?;
    Ok(Self {
        bert,
        dropout,
        classifier,
    })
}
```

修改 `forward` 函數，使用 `pooled_output` 來進行推論。

```rust
// https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1647
pub fn forward(
    &self,
    input_ids: &Tensor,
    token_type_ids: &Tensor,
    attention_mask: Option<&Tensor>,
) -> candle_core::Result<Tensor> {
    let output = self
        .bert
        .forward(input_ids, token_type_ids, attention_mask)?;

    // https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/bert/modeling_bert.py#L1682
    let pooled_output = self.dropout.forward(&output.1.unwrap())?;
    let logits = self.classifier.forward(&pooled_output)?;
    Ok(logits)
}
```

### 3.3 實作 mean_pooling 函數

依官方的 python 程式，移植 `mean_pooling` 函數。其中 `torch.clamp(input_mask_expanded.sum(1), min=1e-9)` 這段程式，我們可以使用 `clamp` 函數來實作。

```rust
pub fn mean_pooling(output: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
    let attention_mask = attention_mask.unsqueeze(candle_core::D::Minus1)?;
    let input_mask_expanded = attention_mask
        .expand(output.shape())?
        .to_dtype(DType::F32)?;
    let sum = output.broadcast_mul(&input_mask_expanded)?.sum(1)?;
    let mask = input_mask_expanded.sum(1)?;
    let mask = mask.clamp(1e-9, f32::INFINITY)?;
    sum / mask
}
```

### 3.4 實作 F.normalize 函數

實作向量正規化函數 `F.normalize`。

```rust
pub fn normalize(t: &Tensor) -> Result<Tensor> {
    let length = t.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?;
    t.broadcast_div(&length)
}
```

### 3.5 推論

與原先的 [cross-encoder](../cross-encoder/main.rs) 程式雷同。依 python 程式，`tokenizer` 需再加 `truncation` 設定。最後再透過 `mean_pooling` 與 `normalize` 函數，得到句子的向量。

```rust
use anyhow::Result;
use candle_core::Tensor;
use macross::{AutoModel, AutoTokenizer};

fn main() -> Result<()> {
    let device = macross::device(false)?;
    let sentences = vec!["This is an example sentence", "Each sentence is converted"];
    let tokenizer = {
        let mut tokenizer =
            AutoTokenizer::from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
                .map_err(anyhow::Error::msg)?;
        let params = tokenizers::PaddingParams::default();
        let truncation = tokenizers::TruncationParams::default();
        let tokenizer = tokenizer.with_padding(Some(params));
        let tokenizer = tokenizer
            .with_truncation(Some(truncation))
            .map_err(anyhow::Error::msg)?;
        tokenizer.clone()
    };

    let bert = macross::models::bert::BertModel::from_pretrained(
        ("sentence-transformers/all-MiniLM-L6-v2", true),
        candle_core::DType::F32,
        &device,
    )?;

    let encoded_input = tokenizer
        .encode_batch(sentences, true)
        .map_err(anyhow::Error::msg)?;

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
    let mean = macross::models::bert::mean_pooling(&result.0, &attention_mask)?;
    let result = macross::normalize(&mean)?;
    println!("result: {:?}", result.to_vec2::<f32>()?);
    Ok(())
}
```

## 4. 復盤

1. 修正原本的 bert 程式。
1. mean-pooling 函數的實作。
1. F.normalize 函數的實作。
