from retriever import FAISSRetriever

# ==========================================
# MOCK DATA
# ==========================================
 
mock_data = [
    {
        "id": 0,
        "text": "VNU (Vietnam National University) was established in 1945."
    },
    {
        "id": 1,
        "text": "The University of Engineering and Technology (UET) is a member of VNU."
    },
    {
        "id": 2,
        "text": "UET is located at 144 Xuan Thuy, Cau Giay, Hanoi."
    },
    {
        "id": 3,
        "text": "Professor Xuan Tu is a famous researcher at VNU UET."
    },
    {
        "id": 4,
        "text": "The annual tuition fee at VNU depends on the specific program."
    }
]

# ==========================================
# BUILD TEMP INDEX
# ==========================================

retriever = FAISSRetriever()

print("Building temporary FAISS index...")

# Vì retriever mới không còn build_index()
# nên ta build trực tiếp ở đây

import faiss
import numpy as np

texts = [x["text"] for x in mock_data]

embeddings = retriever.bi_encoder.encode(
    texts,
    normalize_embeddings=True
)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(
    np.array(embeddings).astype("float32")
)

retriever.index = index
retriever.documents = mock_data

print("Index built successfully.")

# ==========================================
# TEST SEARCH
# ==========================================

query = "Where is UET university located?"

results = retriever.search(
    query=query,
    top_n=5,
    top_k=2
)

print("\n========== TEST RESULTS ==========")
print(f"\nQuestion: {query}\n")

for i, doc in enumerate(results):

    print(f"========== TOP {i+1} ==========")

    print(f"Rerank Score: {doc['rerank_score']:.4f}")

    print(doc["text"])

    print()

# ==========================================
# OPTIONAL SAVE TEST INDEX
# ==========================================

import os
import json

save_dir = "my_vnu_index"

os.makedirs(save_dir, exist_ok=True)

faiss.write_index(
    retriever.index,
    f"{save_dir}/index.faiss"
)

with open(
    f"{save_dir}/documents.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        mock_data,
        f,
        ensure_ascii=False,
        indent=2
    )

print(f"Index saved to: {save_dir}")