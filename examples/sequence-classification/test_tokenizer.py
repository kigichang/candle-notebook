from transformers import BertTokenizerFast

# Paths to your files
vocab_file = "/Users/kigi/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/b2cfda50a1a9fc7919e7444afbb52610d268af92/vocab.txt"
special_tokens_map_file = "/Users/kigi/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/b2cfda50a1a9fc7919e7444afbb52610d268af92/special_tokens_map.json"
tokenizer_config_file = "/Users/kigi/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots/b2cfda50a1a9fc7919e7444afbb52610d268af92/tokenizer_config.json"

# Load the tokenizer
tokenizer = BertTokenizerFast(
    vocab_file=vocab_file,
    special_tokens_map_file=special_tokens_map_file,
    tokenizer_file=None  # Optional: if you don't have a tokenizer.json file
)

# # Optionally update configuration from tokenizer_config.json
# import json

# with open(tokenizer_config_file, "r") as config_file:
#     tokenizer_config = json.load(config_file)
#     tokenizer.init_kwargs.update(tokenizer_config)

# Test the tokenizer
text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

encoded = tokenizer.encode_plus(
    text,
    padding="max_length",
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
print("Encoded:", encoded)
