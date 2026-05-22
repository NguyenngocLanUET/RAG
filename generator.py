import torch

from transformers import pipeline


class QAGenerator: 

    def __init__(
        self,
        model_id="Qwen/Qwen2.5-7B-Instruct"  # Nâng cấp lên bản 7B nếu có thể
    ):

        print("Loading generator model...")

        self.pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            model_kwargs={
                "torch_dtype": torch.float16,
                "load_in_4bit": True
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