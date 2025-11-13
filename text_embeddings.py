import pandas as pd
import requests
import numpy as np
import faiss
from tqdm import tqdm
import time

# === Paths ===
csv_path = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\dataset_chunked.csv"
output_csv_path = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\dataset_with_embeddings.csv"
faiss_index_path = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\faiss_index.bin"

# ollama embedding model
MODEL = "nomic-embed-text"
API_URL = "http://localhost:11434/api/embed"

df = pd.read_csv(csv_path)

def get_embedding(text, retries=3):
    if not isinstance(text, str) or text.strip() == "":
        return np.zeros(768).tolist()

    for attempt in range(retries):
        try:
            response = requests.post(
                API_URL,
                json={"model": MODEL, "input": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            # Ollama returns "embeddings" as list of lists
            if "embeddings" in data:
                return data["embeddings"][0]
            elif "embedding" in data:
                return data["embedding"]
            else:
                raise ValueError(f"Unexpected response: {data}")
        except Exception as e:
            print(f"⚠️  Retry {attempt+1}/{retries} for text: {text[:40]}... due to {e}")
            time.sleep(1 + attempt)
    return np.zeros(768).tolist()

tqdm.pandas(desc="Embedding text")
df["embedding"] = df["text"].progress_apply(get_embedding)

# Save with embeddings
df.to_csv(output_csv_path, index=False)
print(f"✅ Saved embeddings CSV: {output_csv_path}")

# Build FAISS index
embedding_matrix = np.vstack(df["embedding"].values).astype("float32")
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)
faiss.write_index(index, faiss_index_path)

print(f"✅ FAISS index saved to: {faiss_index_path}")
print(f"✅ Total embeddings: {embedding_matrix.shape[0]}, Dimension: {embedding_matrix.shape[1]}")
