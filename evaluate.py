import pandas as pd
from tqdm import tqdm
from retriever import FAISSRetriever
from generator import QAGenerator
import os

def evaluate_rag():
    # 1. Cấu hình đường dẫn
    csv_path = r"d:\rag_system\RAG\uet_qa_dataset_500_with_id.csv"
    index_dir = "my_vnu_index"
    output_report = r"d:\rag_system\RAG\evaluation_report.csv"

    if not os.path.exists(index_dir):
        print(f"Lỗi: Không tìm thấy index tại {index_dir}. Hãy chạy build_index.py trước.")
        return

    # 2. Load dữ liệu kiểm thử
    print("Đang load dataset...")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # 3. Khởi tạo hệ thống
    print("Đang khởi tạo Retriever và Generator...")
    retriever = FAISSRetriever()
    retriever.load(index_dir)
    generator = QAGenerator()

    results = []
    em_count = 0        # Exact Match
    contains_count = 0  # Ground truth contained in prediction

    print(f"Bắt đầu đánh giá {len(df)} câu hỏi...")

    # 4. Chạy đánh giá
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        question = str(row['Question']).strip()
        ground_truth = str(row['Answer']).strip()

        # Retrieval
        retrieved_docs = retriever.search(query=question, top_n=10, top_k=3)

        # Generation
        predicted_answer = generator.get_answer(question, retrieved_docs)

        # So sánh (không phân biệt hoa thường)
        pred_clean = predicted_answer.lower().strip()
        ref_clean = ground_truth.lower().strip()

        is_em = (pred_clean == ref_clean)
        is_contained = (ref_clean in pred_clean) if ref_clean else False

        if is_em: em_count += 1
        if is_contained: contains_count += 1

        results.append({
            "ID": row['ID'],
            "Question": question,
            "Ground Truth": ground_truth,
            "Prediction": predicted_answer,
            "EM": is_em,
            "Contains": is_contained
        })

    # 5. Kết quả tổng quát
    total = len(df)
    print("\n" + "="*40)
    print(f"KẾT QUẢ ĐÁNH GIÁ (N={total}):")
    print(f"- Exact Match Accuracy: {em_count/total:.2%}")
    print(f"- 'Contains GT' Accuracy: {contains_count/total:.2%}")
    print("="*40)

    # Lưu báo cáo chi tiết
    pd.DataFrame(results).to_csv(output_report, index=False, encoding='utf-8-sig')
    print(f"Báo cáo chi tiết đã lưu tại: {output_report}")

if __name__ == "__main__":
    evaluate_rag()