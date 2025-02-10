import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-base')
print('Tokenizer loaded')
print(tokenizer)
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-base')
model.eval()
print('Model loaded')
print(model)
pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
with torch.no_grad():
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    # print("inputs")
    # print(inputs)
    print("inputs.input_ids")
    print(inputs.input_ids.shape)
    print("inputs.attention_mask")
    print(inputs.attention_mask.shape)

    
    scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
    print(scores)
