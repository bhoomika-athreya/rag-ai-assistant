from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example sentences
sentences = [
    "Artificial Intelligence is transforming technology",
    "Machine learning helps computers learn from data",
    "Dogs are domestic animals"
    "What is Artificial Intelligence?",
    "Explain machine learning",
    "How to bake a cake?"
]

# Generate embeddings
embeddings = model.encode(sentences)

# Print results
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding vector size:", len(embedding))
    print()