
# 🚀 Smart-Doc-AI

**Smart-Doc-AI** là ứng dụng hỏi đáp tài liệu thông minh được xây dựng theo kiến trúc **RAG (Retrieval-Augmented Generation)**, kết hợp LLM, truy xuất hybrid, re-ranking và cơ chế tự đánh giá câu trả lời.

Dự án hướng đến giải quyết bài toán: tải lên tài liệu, đặt câu hỏi tự nhiên và nhận câu trả lời có trích dẫn nguồn rõ ràng theo từng trang hoặc phân đoạn nội dung của tài liệu.

## 🎯 Mục tiêu dự án

* **Trả lời chính xác:** Dựa trên nội dung tài liệu thực tế thay vì chỉ dựa vào tri thức sẵn có của mô hình ngôn ngữ.
* **Hỗ trợ đa dạng:** Xử lý tốt cả tài liệu có lớp văn bản (text layer) và tài liệu dạng quét (OCR).
* **So sánh hiệu năng:** Cung cấp hai pipeline xử lý để đối chiếu:
    * **RAG + Self-RAG:** Tự đánh giá và tối ưu truy vấn.
    * **CoRAG (Corrective RAG):** Cơ chế hiệu chỉnh lỗi trong quá trình truy xuất.
* **Minh bạch:** Hiển thị trích dẫn nguồn và thông số đánh giá độ tin cậy giúp người dùng dễ dàng kiểm chứng.

## ✨ Tính năng chính

* **Xử lý đa định dạng:** Tải lên và xử lý tệp PDF, DOCX.
* **Tích hợp OCR:** Tự động nhận diện văn bản cho các trang PDF dạng hình ảnh hoặc không có lớp văn bản.
* **Cấu hình linh hoạt:** Tùy chỉnh các tham số chia nhỏ văn bản (`chunk_size`, `chunk_overlap`) ngay trên giao diện người dùng (UI).
* **Truy xuất Hybrid:** Kết hợp sức mạnh của **BM25** (từ khóa) và **Vector Search** (ngữ nghĩa) qua thư viện FAISS.
* **Reranking:** Sử dụng Cross-encoder để ưu tiên các đoạn văn bản có độ liên quan cao nhất.
* **Self-RAG:**
    * Tự đánh giá chất lượng câu trả lời.
    * Viết lại truy vấn (Query rewrite) khi cần thiết.
    * Truy xuất đa bước (Multi-hop retrieval).
    * Chấm điểm độ tin cậy (Confidence scoring).
* **CoRAG:** Triển khai vòng lặp *truy xuất - tạo - đánh giá - hiệu chỉnh* để cải thiện kết quả khi dữ liệu truy xuất ban đầu chưa tốt.
* **Giao diện trích dẫn (Citation UI):**
    * Hiển thị nguồn dẫn theo từng câu trả lời.
    * Gom nhóm nguồn trong thẻ mở rộng (expander) kèm metadata về trang, tệp gốc và trạng thái OCR.

## 🏗️ Kiến trúc tổng quan

1.  **Ingestion (Tiếp nhận):** Đọc tệp PDF/DOCX và thực hiện OCR nếu cần.
2.  **Processing (Xử lý):** Chia văn bản thành các phân đoạn (chunks), tạo embedding và lưu trữ vào FAISS.
3.  **Retrieval (Truy xuất):** Tìm kiếm kết hợp BM25 và Vector, sau đó tái xếp hạng bằng Cross-encoder.
4.  **Generation (Sinh nội dung):** LLM tạo câu trả lời từ ngữ cảnh. Đánh giá tính xác thực (groundedness) và thực hiện bước hiệu chỉnh nếu cần.
5.  **Presentation (Hiển thị):** Trình bày kết quả từ cả RAG và CoRAG kèm thông tin đánh giá.

## 📂 Cấu trúc thư mục

```text
main.py                # Ứng dụng chính (Streamlit)
modules/ingestion/     # Bộ nạp tài liệu & OCR
modules/processing/    # Bộ chia nhỏ văn bản (Text splitter)
modules/embedding/     # Mô hình nhúng dữ liệu
modules/vectorstore/   # FAISS & Bộ truy xuất Hybrid
modules/rag/           # Logic LLM, pipeline Self-RAG/CoRAG & Trích dẫn
tests/                 # Các bản kiểm thử đơn vị (Unit tests)
Makefile               # Các lệnh cài đặt và chạy nhanh
```

## 🛠️ Cài đặt nhanh

### Cách 1: Sử dụng Makefile (Khuyên dùng)
```bash
make run
```
Lệnh này sẽ tự động: tạo môi trường ảo (`venv`), cài đặt các thư viện cần thiết và khởi chạy ứng dụng Streamlit.

### Cách 2: Cài đặt thủ công
```bash
# Tạo và kích hoạt môi trường ảo
python3 -m venv venv
source venv/bin/activate  # Trên Windows dùng: venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt

# Chạy ứng dụng
streamlit run main.py
```

## ⚙️ Cấu hình môi trường

Tạo file `.env` tại thư mục gốc và cấu hình các thông số sau:

```env
GROQ_API_KEY=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
```

## ⚠️ Yêu cầu hệ thống cho OCR

Để tính năng OCR hoạt động ổn định, máy tính của bạn cần cài đặt sẵn:
* `tesseract-ocr`
* `poppler-utils`

## 🧪 Chạy kiểm thử (Testing)

```bash
python -m unittest tests.test_rag_pipeline
```

## 📝 Ghi chú

Dự án này được phát triển cho mục đích học thuật và thực nghiệm ứng dụng RAG. Kết quả đầu ra phụ thuộc rất lớn vào chất lượng tài liệu đầu vào, cấu hình chia nhỏ văn bản (chunking) và mô hình ngôn ngữ (LLM) đang sử dụng.
