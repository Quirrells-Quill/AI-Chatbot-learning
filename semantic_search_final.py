"""
Optimized Semantic Search with FAISS (Cosine Similarity)
---------------------------------------------------------
- Loads or builds FAISS index automatically (flat or IVF)
- Normalizes embeddings for cosine similarity (fixes same-results issue)
- Uses inner-product FAISS index for cosine search
- Fast startup using .npy cached embeddings
- Supports subset search
"""

import os
import json
import sys
import traceback
import numpy as np
import pandas as pd
import faiss
import requests
import time
from typing import Optional, List

# ====== CONFIG ====== #
OLLAMA_BASE = "http://localhost:11434/api"
MODEL = "nomic-embed-text"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3

try:
    # Python 3.7+: reconfigure the text encoding of stdout/stderr
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # fallback for older Pythons or unusual streams
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        # last resort: do nothing (we'll still try to avoid crash)
        pass

CSV_PATH = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\dataset_with_embeddings.csv"
EMB_NPY_PATH = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\embeddings.npy"
TEXTS_CSV_PATH = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\texts.csv"
FAISS_INDEX_PATH = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\faiss_index.bin"

FULL_INDEX_MODE = "ivf"  # or "flat"
N_LIST = 256
N_PROBE = 16
# ==================== #


# ====== Load or Prepare Embeddings ====== #
def load_or_prepare_embeddings(csv_path=CSV_PATH, emb_npy=EMB_NPY_PATH, texts_csv=TEXTS_CSV_PATH, force_reload=False):
    if (not force_reload) and os.path.exists(emb_npy) and os.path.exists(texts_csv):
        print("✅ Loading embeddings from .npy and texts from CSV...")
        embeddings = np.load(emb_npy)
        df_texts = pd.read_csv(texts_csv)
        if len(df_texts) != embeddings.shape[0]:
            raise ValueError("Mismatch between texts.csv rows and embeddings.npy rows.")
        # Normalize for cosine similarity
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
        return df_texts, embeddings.astype('float32')

    print("📦 Reading CSV and parsing embeddings...")
    try:
        df = pd.read_csv(csv_path, usecols=['text', 'embedding'])
    except Exception:
        df = pd.read_csv(csv_path)

    def parse_emb(s):
        if isinstance(s, (list, np.ndarray)):
            return np.array(s, dtype='float32')
        return np.array(json.loads(s), dtype='float32')

    df['embedding'] = df['embedding'].map(parse_emb)
    embeddings = np.vstack(df['embedding'].values).astype('float32')

    # Normalize embeddings
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10

    np.save(emb_npy, embeddings)
    df_texts = df.drop(columns=['embedding'])
    df_texts.to_csv(texts_csv, index=False)
    print(f"✅ Saved embeddings -> {emb_npy}\n✅ Saved texts -> {texts_csv}")
    return df_texts, embeddings


# ====== Build or Load FAISS Index ====== #
def build_or_load_full_index(embeddings: np.ndarray, index_path=FAISS_INDEX_PATH, mode: str = FULL_INDEX_MODE, n_list: int = N_LIST):
    d = embeddings.shape[1]
    if os.path.exists(index_path):
        try:
            idx = faiss.read_index(index_path)
            if idx.d != d:
                raise ValueError("Dimension mismatch in saved index.")
            print("✅ Loaded FAISS index from disk.")
            return idx
        except Exception as e:
            print("⚠️ Failed to load existing index. Rebuilding...", e)

    print("⚙️ Building FAISS index (this may take a while)...")

    if mode == "flat":
        index = faiss.IndexFlatIP(d)  # cosine similarity
        index.add(embeddings)
    elif mode == "ivf":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_list, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            print("🔧 Training IVF index...")
            index.train(embeddings)
        index.add(embeddings)
    else:
        raise ValueError("Unknown mode. Use 'ivf' or 'flat'.")

    faiss.write_index(index, index_path)
    print(f"✅ FAISS index saved to {index_path}")
    return index


# ====== Embedding Helper Functions ====== #
def call_ollama_embed_single(text):
    payload = {"model": MODEL, "input": text}
    url1 = f"{OLLAMA_BASE}/embed"
    url2 = f"{OLLAMA_BASE}/embeddings"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url1, json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception:
            try:
                r = requests.post(url2, json={"model": MODEL, "prompt": text}, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(0.5 * attempt)


def get_embedding_resilient(text: str):
    if not isinstance(text, str) or text.strip() == "":
        return np.zeros(768, dtype='float32')
    j = call_ollama_embed_single(text)
    if "embedding" in j:
        emb = np.asarray(j["embedding"], dtype='float32')
    elif "embeddings" in j:
        emb = np.asarray(j["embeddings"][0], dtype='float32')
    elif "data" in j and isinstance(j["data"], list):
        emb = np.asarray(j["data"][0]["embedding"], dtype='float32')
    else:
        raise RuntimeError(f"Unexpected embedding structure: {j}")
    return emb


def embed_query(query: str, get_embedding_fn) -> np.ndarray:
    q_emb = np.array(get_embedding_fn(query), dtype='float32')
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10
    return q_emb


# ====== Search Functions ====== #
def search_full_index(query: str, index, df_texts: pd.DataFrame, get_embedding_fn, k: int = 5, nprobe: Optional[int] = N_PROBE):
    if hasattr(index, 'nprobe'):
        index.nprobe = nprobe
    q = embed_query(query, get_embedding_fn)
    D, I = index.search(q, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        row = df_texts.iloc[idx].to_dict()
        row['_score'] = float(dist)
        row['_index'] = int(idx)
        results.append(row)
    return results


def search_filtered_subset(query: str, df_texts: pd.DataFrame, embeddings: np.ndarray,
                           filter_mask: Optional[pd.Series] = None,
                           sample_frac: Optional[float] = None,
                           get_embedding_fn=None,
                           k: int = 5) -> List[dict]:
    if filter_mask is not None:
        subset_idx = np.flatnonzero(filter_mask.values)
    elif sample_frac is not None:
        rng = np.random.default_rng(42)
        n = len(df_texts)
        choose = rng.choice(np.arange(n), size=max(1, int(n * sample_frac)), replace=False)
        subset_idx = np.sort(choose)
    else:
        raise ValueError("Provide filter_mask or sample_frac for subset search.")

    sub_emb = embeddings[subset_idx].astype('float32')
    sub_index = faiss.IndexFlatIP(sub_emb.shape[1])
    sub_index.add(sub_emb)

    q = embed_query(query, get_embedding_fn)
    D, I = sub_index.search(q, k)
    results = []
    for dist, rel_idx in zip(D[0], I[0]):
        if rel_idx < 0:
            continue
        global_idx = int(subset_idx[rel_idx])
        row = df_texts.iloc[global_idx].to_dict()
        row['_score'] = float(dist)
        row['_index'] = int(global_idx)
        results.append(row)
    return results


# ====== Main ====== #
if __name__ == "__main__":
    
    print("=== semantic_search_final.py starting ===")
    print("sys.executable:", sys.executable)
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
    print("cwd:", os.getcwd())
    print("argv:", sys.argv)
    sys.stdout.flush()
    # Wrap main flow so exceptions are visible + non-zero exit
    try:
        # ----------------------------
        # (Keep the rest of your original script here)
        # For example the entire existing main section should be indented or
        # moved into a function and called here. If you don't want to restructure,
        # at minimum add a try/except around parts that may throw:
        # ----------------------------
        pass  # <- replace pass with actual main body or ensure main() is wrapped
    except Exception as _e:
        print("=== semantic_search_final.py ERROR ===", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
    start_time = time.time()

    print("🚀 Loading dataset and embeddings...")
    df_texts, embeddings = load_or_prepare_embeddings(CSV_PATH, EMB_NPY_PATH, TEXTS_CSV_PATH, force_reload=False)

    print("⚙️ Building or loading FAISS index...")
    full_index = build_or_load_full_index(embeddings, FAISS_INDEX_PATH, mode=FULL_INDEX_MODE, n_list=N_LIST)

    get_embedding = get_embedding_resilient

    query_file = r"C:\Users\khush\OneDrive\Desktop\AI project\query.txt"
    output_path = r"C:\Users\khush\OneDrive\Desktop\AI project\Temp_semantic.txt"

    try:
        with open(query_file, "r", encoding="utf-8") as f:
            query = f.read().strip()

        print(f"\n🔍 Searching for: '{query}'")
        results = search_full_index(query, full_index, df_texts, get_embedding_fn=get_embedding, k=5, nprobe=N_PROBE)

        print(f"\n✅ Search completed in {time.time() - start_time:.3f} seconds\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. Score: {r['_score']:.4f}\n{r.get('text', '')[:300]}\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Semantic Search Results\nQuery: {query}\n\n")
            for i, r in enumerate(results, start=1):
                f.write(f"{i}. Score: {r['_score']:.4f}\n{r.get('text', '')}\n\n")

        print(f"📄 Results saved to: {output_path}")

    except Exception as e:
        print(f"❌ Search failed: {e}")

