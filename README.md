# Hệ thống mini RAG (Retriever and Generator) sử dụng dữ liệu liên quan đến tin tức của trường Đại học công nghệ (UET)
**Tái lập kết quả**:
```
Thứ tự chạy các ipynb: `generate-data`(1) -> `rag-system`(2) -> `evaluate`(3)
Sau khi save version để chạy và tạo kết quả tự động từ các ipynb, thêm trực tiếp ipynb đó vào ipynb tiếp theo để tránh phải tải lại dữ liệu đã sinh
Cụ thể: sau khi chạy (1), mở (2) và add ipynb (1), sau khi chạy (2), mở (3) và add ipynb (2).
```
**Các chức năng**:
- (1) tạo dữ liệu giả (**IAA=91.5%**): 
```
+ Web crawling và trích xuất nội dung bài viết: requests, beautifulSoup
+ Rule-based text cleaning
+ Prompt engineering để sinh câu hỏi bằng `Qwen2.5-7B-Instruct`. Prompt được chỉnh để tạo ra câu hỏi mang khái quát thông tin để có thể đứng độc lập và câu trả lời ngắn gọn. Ví dụ: KHÔNG: `Học bổng trị giá bao nhiêu?` MÀ: `Học bổng Mitsubishi năm 2023 tại UET trị giá bao nhiêu?`.
+ Batch inference tối ưu.
```
- (2) xây dựng hệ thống Hybrid Retriever (bi_encoder_name=`BAAI/bge-m3`, reranker_name=`BAAI/bge-reranker-base`, tokenizer=`bm25`) và Generator (Qwen/Qwen2.5-7B-Instruct). Build index bằng nội dung được crawl và làm sạch từ (1).
- (3) đánh giá hệ thống RAG với 400 câu hỏi đã được chuẩn bị sẵn từ (1). 
