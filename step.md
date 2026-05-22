# Hướng dẫn xử lý tập dữ liệu UET QA

Để tích hợp file `uet_qa_dataset_500_with_id.csv` vào hệ thống RAG, bạn hãy thực hiện theo các bước sau:

1. **Tiền xử lý dữ liệu (Data Preprocessing):**
   - Sử dụng thư viện `pandas` để load file với encoding `utf-8-sig`.
   - Làm sạch dữ liệu: Loại bỏ các ký tự đặc biệt thừa và chuẩn hóa khoảng trắng trong cột `Question` và `Answer`.

2. **Tạo Vector Database (Indexing):**
   - Sử dụng cột `Answer` (kết hợp với `Title`) để làm văn bản ngữ cảnh (Context).
   - Sử dụng một Embedding Model phù hợp với tiếng Việt (ví dụ: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` hoặc các model của VinAI).
   - Lưu trữ các vector vào database như FAISS, ChromaDB hoặc Pinecone.

3. **Tích hợp với Generator:**
   - Khi nhận một câu hỏi (`Question`), thực hiện tìm kiếm các đoạn văn bản liên quan nhất từ Vector DB.
   - Truyền danh sách các đoạn văn bản này vào hàm `get_answer` của class `QAGenerator` trong file `generator.py`.
   - So sánh câu trả lời sinh ra với cột `Answer` trong file CSV để đánh giá độ chính xác (Ground Truth).