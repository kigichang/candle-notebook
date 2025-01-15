from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"

sentences = ["What is BGE M3?", "Defination of BM25"]

model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(sentences)
print(embeddings)
