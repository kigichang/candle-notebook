from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]

MODEL_NAME = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).eval()

texts = ["What is BGE M3?", "Defination of BM25"]

# 文本處理
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# 獲取嵌入
with torch.no_grad():
    outputs = model(**inputs)
    print("outputs")
    print(outputs)

sentence_embeddings = cls_pooling(outputs, inputs['attention_mask'])
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
print("Sentence embeddings:")
print(sentence_embeddings)