import os
import json
import faiss
import pandas as pd
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
 

EMBED_MODEL_NAME = "BAAI/bge-m3"


def process_csv_to_chunks(csv_path):
    df = pd.read_csv(csv_path)
    # Chuyển đổi mỗi dòng thành một chuỗi văn bản mô tả
    cleaned_chunks = []
    
    for idx, row in df.iterrows():
        # Tạo chuỗi văn bản từ tất cả các cột của dòng đó
        # Ví dụ: "Cột A: Giá trị A, Cột B: Giá trị B..."
        row_text = ". ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
        
        cleaned_chunks.append({
            "id": idx,
            "text": row_text,
            "metadata": row.to_dict()
        })
    
    return cleaned_chunks


def build_embeddings(chunks):
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]

    # cosine similarity
    index = faiss.IndexFlatIP(dim)

    index.add(
        np.array(embeddings).astype("float32")
    )

    return index


def save_index(index, chunks, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    faiss.write_index(index, f"{output_dir}/index.faiss")

    with open(f"{output_dir}/documents.json", "w", encoding="utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved {len(chunks)} chunks.")


def build_pipeline(raw_data_path, output_dir):
    print(f"Loading and processing CSV: {raw_data_path}...")
    chunks = process_csv_to_chunks(raw_data_path)

    print(f"Total chunks: {len(chunks)}")

    print("Building embeddings...")
    embeddings = build_embeddings(chunks)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Saving...")
    save_index(index, chunks, output_dir)

    print("Done.")


if __name__ == "__main__":
    build_pipeline(
        raw_data_path="data.csv",
        output_dir="my_vnu_index"
    )