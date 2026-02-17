import fitz

def extract_text_from_pdf(pdf_path):
    pages = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "text": text,
            "page": page_num + 1,
            "source": pdf_path
        })
    doc.close()
    return pages