from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
print(tokenizer)
# 存 tokenizer
tokenizer.save_pretrained("tmp") # 匯出檔案至 tmp 目錄下。