import json
import faiss
import numpy as np
import sys
import os
sys.path.append('/kaggle/working/RAG')
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)


class FAISSRetriever:

    def __init__(
        self,
        bi_encoder_name="BAAI/bge-m3",
        reranker_name="BAAI/bge-reranker-base"
    ):
 
        print("Loading embedding model...")
        self.bi_encoder = SentenceTransformer(
            bi_encoder_name
        )

        print("Loading reranker...")
        self.reranker = CrossEncoder(
            reranker_name
        )

        self.index = None
        self.documents = []

    def load(self, index_dir):
 
        self.index = faiss.read_index(
            f"{index_dir}/index.faiss"
        )

        with open(
            f"{index_dir}/documents.json",
            "r",
            encoding="utf-8"
        ) as f:

            self.documents = json.load(f)

        print(f"Loaded {len(self.documents)} documents.")

    def retrieve(
        self,
        query,
        top_n=20
    ):

        query_embedding = self.bi_encoder.encode(
            [query],
            normalize_embeddings=True
        )

        scores, indices = self.index.search(
            np.array(query_embedding).astype("float32"),
            top_n
        )

        candidates = []

        for idx, score in zip(indices[0], scores[0]):

            if idx == -1:
                continue

            candidates.append({
                "text": self.documents[idx]["text"],
                "retrieval_score": float(score)
            })

        return candidates

    def rerank(
        self,
        query,
        candidates,
        top_k=3
    ):

        pairs = [
            [query, c["text"]]
            for c in candidates
        ]

        rerank_scores = self.reranker.predict(
            pairs,
            batch_size=16
        )

        ranked = []

        for cand, score in zip(candidates, rerank_scores):

            ranked.append({
                "text": cand["text"],
                "rerank_score": float(score)
            })

        ranked = sorted(
            ranked,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return ranked[:top_k]

    def search(
        self,
        query,
        top_n=20,
        top_k=3
    ):

        candidates = self.retrieve(
            query=query,
            top_n=top_n
        )

        reranked = self.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k
        )

        return reranked