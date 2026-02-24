import fitz # PyMuPDF
from docx import Document
import os


def extract_text_from_pdf(file_path:str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page_num  , page in enumerate(doc):
        text += f"\n----- Page {page_num+1} -----\n"
        text += page.get_text()
    doc.close()
    return text

def extract_text_from_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text + "\n"
    return text

def extract_text(file_path: str)-> str:
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f'Unsupported file type: {ext}')
    
