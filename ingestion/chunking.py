import re

def clean_text(text):
    text = re.sub(r"\s+"," ",text)
    text = re.sub(r"-\s+","",text)
    return text.strip()

def chunk_pages(pages, chunk_size=500, overlap=50):
    chunks = []
    for page_data in pages:
        words = page_data["text"].split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append({
                "text": " ".join(chunk_words),
                "page": page_data["page"],
                "source": page_data["source"]
            })
    return chunks
