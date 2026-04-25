# Smart-Doc-AI

Smart-Doc-AI la ung dung hoi dap tai lieu thong minh duoc xay dung theo kien truc RAG (Retrieval-Augmented Generation), ket hop LLM, truy xuat hybrid, re-ranking va co che tu danh gia cau tra loi.

Du an huong den bai toan: upload tai lieu, dat cau hoi tu nhien, nhan cau tra loi co trich dan nguon ro rang theo tung trang/noi dung tai lieu.

## Muc tieu du an

- Tra loi cau hoi dua tren noi dung tai lieu thay vi tri nho mo hinh.
- Ho tro tai lieu co text layer va tai lieu scan (OCR).
- Cung cap so sanh hai pipeline:
	- RAG + Self-RAG
	- CoRAG (Corrective RAG)
- Hien thi trich dan va thong tin danh gia do tin cay de nguoi dung de kiem chung.

## Tinh nang chinh

- Upload va xu ly tai lieu PDF, DOCX.
- OCR tu dong cho trang PDF co hinh anh/khong co text layer.
- Chia van ban thanh chunks voi tham so `chunk_size`, `chunk_overlap` co the cau hinh tren UI.
- Hybrid retrieval: BM25 + Vector Search (FAISS).
- Cross-encoder reranking de uu tien doan van ban lien quan hon.
- Self-RAG:
	- Tu danh gia chat luong cau tra loi
	- Query rewrite khi can
	- Multi-hop retrieval
	- Confidence scoring
- CoRAG:
	- Vong corrective retrieve/generate/evaluate de cai thien cau tra loi khi ket qua chua du tot.
- Citation UI:
	- Trich dan theo tung cau tra loi (RAG/CoRAG)
	- Gom nguon trong expander, co metadata trang/nguon/OCR.

## Kien truc tong quan

1. Ingestion
- Doc PDF/DOCX.
- OCR neu can.

2. Processing
- Split tai lieu thanh chunks.
- Tao embedding va luu vao FAISS.

3. Retrieval
- Truy xuat ket hop BM25 + vector.
- Rerank bang cross-encoder.

4. Generation
- Sinh cau tra loi tu context da retrieve.
- Danh gia groundedness, confidence, va thu corrective pass neu can.

5. Presentation
- Hien thi ket qua RAG va CoRAG.
- Hien thi citations va thong tin evaluation.

## Cau truc thu muc

```text
main.py                      # Streamlit app
modules/ingestion/           # Loader + OCR
modules/processing/          # Text splitter
modules/embedding/           # Embedding model
modules/vectorstore/         # FAISS + Hybrid retriever
modules/rag/                 # LLM, pipeline Self-RAG/CoRAG, citation
tests/                       # Unit tests
Makefile                     # Lenh cai dat/chay nhanh
```

## Cai dat nhanh

### Cach 1: dung Makefile (khuyen nghi)

```bash
make run
```

Lenh nay se:
- Tao `venv` neu chua co
- Cai dependencies tu `requirements.txt`
- Chay Streamlit app

### Cach 2: thu cong

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

## Cau hinh moi truong

Tao file `.env` (hoac export bien moi truong) voi toi thieu:

```env
GROQ_API_KEY=your_api_key
GROQ_MODEL=llama-3.1-8b-instant
```

`GROQ_MODEL` la tuy chon, neu bo trong he thong se dung mac dinh.

## Yeu cau he thong cho OCR

De OCR hoat dong on dinh, may can cai dat:

- `tesseract-ocr`
- `poppler-utils`

Neu chua co, OCR cho PDF scan co the khong cho ket qua nhu mong doi.

## Chay test

```bash
python -m unittest tests.test_rag_pipeline
```

## Dinh huong su dung

- Chon tai lieu PDF/DOCX
- Dat cau hoi theo noi dung tai lieu
- Kiem tra cau tra loi va doi chieu nguon trich dan
- So sanh chat luong giua RAG va CoRAG de danh gia hieu qua corrective loop

## Ghi chu

Day la du an hoc thuat/ung dung RAG thuc nghiem. Ket qua phu thuoc vao chat luong tai lieu dau vao, cau hinh chunking, retrieval va model dang su dung.
