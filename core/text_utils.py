import re

def clean_transcript(text):
  text = re.sub(r'\b(uh|um|ah|mm|oh|like|you know)\b', '', text, flags=re.IGNORECASE)
  text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)
  text = re.sub(r'\b(Speaker \d+|[A-Z ]+:)\b', '', text, flags=re.IGNORECASE)
  text = re.sub(r'[^\w\s.,?!]', '', text)
  text = re.sub(r'\s+', ' ', text).strip()
  text = re.sub(r'[.]{2,}', '.', text)
  return text

def approximate_token_count(text):
  return len(text) // 3

def chunk_text(text, max_tokens):
  chunks, current = [], ""
  for sentence in re.split(r'(?<=[.!?]) +', text):
    if approximate_token_count(current + sentence) < max_tokens:
      current += sentence + " "
    else:
      chunks.append(current.strip())
      current = sentence + " "
  if current:
    chunks.append(current.strip())
  return chunks