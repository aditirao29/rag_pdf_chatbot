import re

def clean_text(text):
    text = re.sub(r"\s+"," ",text)
    text = re.sub(r"-\s+","",text)
    return text.strip()

def chunk_text(text,chunk_size=500,overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start<len(words):
        end = start+chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start+=chunk_size-overlap
    return chunks
