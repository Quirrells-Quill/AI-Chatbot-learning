import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# === Paths ===
folder = Path(r"C:\Users\khush\OneDrive\Desktop\AI project\Customized training data")
output_folder = folder / "csv_files_for_the_training_data"
output_folder.mkdir(exist_ok=True)

raw_csv_path = output_folder / "dataset_raw.csv"

# === Extraction ===
print("📄 Extracting text from PDF and TXT files...")
records = []

pdf_files = list(folder.glob("*.pdf"))
txt_files = list(folder.glob("*.txt"))
total_files = len(pdf_files) + len(txt_files)

for pdf_path in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
    doc = fitz.open(pdf_path)
    text = "".join([page.get_text("text") for page in doc])
    records.append({
        "source": pdf_path.name,
        "title": pdf_path.stem,
        "category": "PDF",
        "text": text.strip()
    })

for txt_path in tqdm(txt_files, desc="Processing TXTs", unit="file"):
    with open(txt_path, "r", encoding="utf-8") as f:
        records.append({
            "source": txt_path.name,
            "title": txt_path.stem,
            "category": "TXT",
            "text": f.read().strip()
        })

df = pd.DataFrame(records)
df.to_csv(raw_csv_path, index=False, encoding="utf-8")

print(f"\n✅ Saved raw dataset: {raw_csv_path}")
print(f"✅ Total files processed: {total_files}")
