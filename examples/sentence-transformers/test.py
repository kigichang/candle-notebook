import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

input_texts = [
    "what is the capital of China?",
    "how to implement quick sort in python?",
    "北京",
    "快排算法介绍"
]

model_name_or_path = 'Alibaba-NLP/gte-multilingual-base'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
print(tokenizer)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
print(model)

# Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

outputs = model(**batch_dict)

dimension=768 # The output dimension of the output embedding, should be in [128, 768]
embeddings = outputs.last_hidden_state[:, 0][:dimension]

embeddings = F.normalize(embeddings, p=2, dim=1)
print(embeddings.shape)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores.tolist())
