from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizerFast
import torch

model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
#print(tokenizer)
tokenizer.save_pretrained("tmp")
features = tokenizer(
    [
        'How many people live in Berlin?', 
        'How many people live in Berlin?'
    ], 
    [
        'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.', 
        'New York City is famous for the Metropolitan Museum of Art.'
    ],
    padding=True, truncation=True, return_tensors="pt")



print("input_ids", features.input_ids.shape)
print("token_type_ids", features.token_type_ids.shape)
print("attention_mask", features.attention_mask.shape)

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    print(scores)
