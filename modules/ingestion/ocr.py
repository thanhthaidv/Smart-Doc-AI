from typing import List

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image


def detect_pdf_has_text(file_path: str, min_chars: int = 20, max_pages: int = 2) -> bool:
	"""Trả về True nếu tệp PDF có vẻ chứa văn bản có thể chọn."""
	try:
		with pdfplumber.open(file_path) as pdf:
			for page in pdf.pages[:max_pages]:
				text = page.extract_text() or ""
				if len(text.strip()) >= min_chars:
					return True
	except Exception:
		return False
	return False


def detect_pdf_has_images(file_path: str, max_pages: int = 2) -> bool:
	"""Trả về True nếu tệp PDF có vẻ chứa hình ảnh."""
	try:
		with pdfplumber.open(file_path) as pdf:
			for page in pdf.pages[:max_pages]:
				if page.images:
					return True
	except Exception:
		return False
	return False


def ocr_image_to_text(image: Image.Image, lang: str = "vie+eng") -> str:
	"""Chạy chương trình OCR trên ảnh PIL và trả về văn bản đã trích xuất."""
	return pytesseract.image_to_string(image, lang=lang)


def ocr_pdf_to_text(file_path: str, lang: str = "vie+eng", dpi: int = 300) -> str:
	"""Chuyển đổi các trang PDF thành hình ảnh và chạy OCR để tạo thành một chuỗi đơn."""
	pages = convert_from_path(file_path, dpi=dpi)
	texts: List[str] = []
	for page in pages:
		texts.append(ocr_image_to_text(page, lang=lang))
	return "\n".join(texts)
