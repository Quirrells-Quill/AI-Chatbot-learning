import os
import json
import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- CONFIG ----------
CSV_PATH = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\dataset_chunked.csv"
OUT_CSV = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\dataset_with_embeddings.csv"
EMB_NPY = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\embeddings.npy"
FAISS_INDEX_PATH = r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data\faiss_index.bin"

OLLAMA_BASE = "http://localhost:11434/api"   # Ollama default base
MODEL = "nomic-embed-text"                   # your embedding model (must be pulled)
NUM_WORKERS = 8                              # adjust to your CPU / disk / Ollama capacity
BATCH_SAVE_EVERY = 500                       # persist progress after this many items
REQUEST_TIMEOUT = 30                         # seconds per request
MAX_RETRIES = 3
# ----------------------------

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

def try_post(url, payload, timeout=REQUEST_TIMEOUT):
    """Post with retries; returns JSON or raises."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep(0.5 * attempt)

def get_embedding_single(text: str):
    """
    Robust single-text embedding call:
    - Tries /api/embed with 'input' (common)
    - Fallback to /api/embeddings with 'prompt' (some model READMEs)
    Returns: Python list of floats
    """
    if not isinstance(text, str) or text.strip() == "":
        # if you expect a different dim, change fallback zeros accordingly
        return np.zeros(768).tolist()

    # Try /api/embed first (many Ollama examples use this with "input")
    payload1 = {"model": MODEL, "input": text}
    try:
        res = try_post(f"{OLLAMA_BASE}/embed", payload1)
        # some endpoints return {"embedding": [...]} or {"data":[{"embedding":[...]}]}
        if isinstance(res, dict):
            if "embedding" in res:
                return res["embedding"]
            if "data" in res and isinstance(res["data"], list) and "embedding" in res["data"][0]:
                return res["data"][0]["embedding"]
        # fallback: if we got something unexpected, keep going to next
    except Exception:
        pass

    # Fallback to /api/embeddings (some models/docs use this with "prompt")
    payload2 = {"model": MODEL, "prompt": text}
    res2 = try_post(f"{OLLAMA_BASE}/embeddings", payload2)
    # Expected: {"embedding": [...]} per model README for nomic-embed-text
    if "embedding" in res2:
        return res2["embedding"]
    # other shapes:
    if "data" in res2 and isinstance(res2["data"], list) and "embedding" in res2["data"][0]:
        return res2["data"][0]["embedding"]

    raise RuntimeError("Unexpected embedding response shape")

def embed_worker(i_text):
    idx, text = i_text
    try:
        emb = get_embedding_single(text)
        return idx, np.asarray(emb, dtype="float32")
    except Exception as e:
        # on error, return None (caller decides what to do)
        print(f"[error] idx {idx}: {e}")
        return idx, None

def main():
    df = pd.read_csv(CSV_PATH)
    n = len(df)
    print(f"Loaded {n} rows from {CSV_PATH}")

    # if we already have embeddings saved, load and reuse
    embeddings = None
    if os.path.exists(EMB_NPY):
        embeddings = np.load(EMB_NPY)
        if embeddings.shape[0] != n:
            print("Existing embeddings.npy length mismatch -> ignoring existing file.")
            embeddings = None
        else:
            print("Loaded embeddings from", EMB_NPY)

    # Create empty embedding matrix if none exists
    if embeddings is None:
        # try to infer dimension from first call? we'll fill as we go; use list then stack
        embeddings_list = [None] * n

        items = list(enumerate(df["text"].astype(str).tolist()))

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(embed_worker, it): it for it in items}
            completed = 0
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
                idx, emb = fut.result()
                if emb is None:
                    # fallback zero vector; adjust dimension if you know it
                    emb = np.zeros(768, dtype="float32")
                embeddings_list[idx] = emb
                completed += 1

                # save intermittently
                if completed % BATCH_SAVE_EVERY == 0:
                    arr = np.vstack([x for x in embeddings_list]).astype("float32")
                    np.save(EMB_NPY, arr)
                    print(f"Checkpoint saved ({completed}/{n}) -> {EMB_NPY}")

        embeddings = np.vstack(embeddings_list).astype("float32")
        np.save(EMB_NPY, embeddings)
        print("Saved embeddings to", EMB_NPY)

    # attach embeddings to dataframe as list strings (optional)
    # convert to python lists to allow CSV serialization
    df["embedding"] = df.index.map(lambda i: embeddings[i].tolist())

    # save csv once
    df.to_csv(OUT_CSV, index=False)
    print("Wrote CSV with embeddings ->", OUT_CSV)

    import faiss
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Saved FAISS index ->", FAISS_INDEX_PATH)

if __name__ == "__main__":
    main()
