from typing import List
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

def detect_pdf_has_text(file_path: str, min_chars: int = 20, max_pages: int = 5) -> bool:
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages[:max_pages]
            for page in pages:
                if (page.extract_text() or "").strip() and len(page.extract_text().strip()) >= min_chars:
                    return True
    except: return False
    return False

def detect_pdf_has_images(file_path: str, max_pages: int = 100) -> bool:
    try:
        with pdfplumber.open(file_path) as pdf:
            pages = pdf.pages[:max_pages]
            for page in pages:
                if page.images: return True
    except: return False
    return False

def ocr_image_to_text(image: Image.Image, lang: str = "vie+eng") -> str:
    try:
        # psm 1 giúp nhận diện cấu trúc trang tốt hơn (thơ, cột)
        custom_config = r'--oem 3 --psm 1'
        return pytesseract.image_to_string(image, lang=lang, config=custom_config)
    except: return ""

def ocr_pdf_pages_to_text(file_path: str, lang: str = "vie+eng", dpi: int = 300) -> List[str]:
    try:
        # Lưu ý: convert_from_path yêu cầu cài đặt poppler
        pages = convert_from_path(file_path, dpi=dpi)
        return [ocr_image_to_text(p, lang) for p in pages]
    except: return []