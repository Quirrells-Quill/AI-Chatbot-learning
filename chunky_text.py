import pandas as pd
import textwrap
from tqdm import tqdm
from pathlib import Path

# === Paths ===
folder = Path(r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data\csv_files_for_the_training_data")
raw_csv_path = folder / "dataset_raw.csv"
chunked_csv_path = folder / "dataset_chunked.csv"

print("✂️ Chunking text into smaller pieces...")

df = pd.read_csv(raw_csv_path)
chunks = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking text", unit="doc"):
    for chunk in textwrap.wrap(row["text"], 1000):
        chunks.append({
            "source": row["source"],
            "title": row["title"],
            "category": row["category"],
            "text": chunk
        })

chunked_df = pd.DataFrame(chunks)
chunked_df.to_csv(chunked_csv_path, index=False, encoding="utf-8")

print(f"\n✅ Saved chunked dataset: {chunked_csv_path}")
print(f"✅ Total chunks created: {len(chunked_df)}")
