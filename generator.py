import torch
from transformers import pipeline, BitsAndBytesConfig

import sys
import os
sys.path.append('/kaggle/working/RAG')
class QAGenerator: 

    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        print("Loading generator model...")

        # Cấu hình 4-bit chính xác
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "quantization_config": quantization_config,
                "device_map": "auto"
            }
        )

    def build_prompt(
        self,
        question,
        contexts
    ):

        context_text = "\n\n".join([
            f"[Tài liệu {i+1}]\n{c}"
            for i, c in enumerate(contexts)
        ])

        prompt = f"""
You are a factual Vietnamese QA assistant.
Dựa vào các thông tin (Context) được cung cấp dưới đây, hãy trả lời câu hỏi một cách chính xác.

Lưu ý quan trọng:
- CHỈ sử dụng thông tin trong Context.
- Nếu không có thông tin trong Context, hãy trả lời:
"I don't know"
- Câu trả lời cần ngắn gọn, đúng trọng tâm.

Context:
{context_text}

Question:
{question}

Trả lời:
"""

        return prompt

    def get_answer(
        self,
        question,
        retrieved_docs
    ):

        if len(retrieved_docs) == 0:
            return "I don't know"

        contexts = [
            d["text"]
            for d in retrieved_docs
        ]

        prompt = self.build_prompt(
            question,
            contexts
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=40,
            do_sample=False,
            return_full_text=False
        )

        answer = outputs[0]["generated_text"].strip()

        answer = answer.split("\n")[0]
        answer = answer.strip()

        return answer