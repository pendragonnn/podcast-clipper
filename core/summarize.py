from llama_cpp import Llama
from .text_utils import chunk_text
import os

def generate_summary_with_llama(text, output_name="summary.txt"):
  llm = Llama(model_path="./llama-2-7b-chat.Q4_K_M.gguf", n_gpu_layers=-1)
  chunk_token_limit = 256
  chunks = chunk_text(text, chunk_token_limit)
  summaries = []
  for chunk in chunks:
    prompt = f"""
    You are an expert summarizer. Summarize the following for a short video:
      {chunk}
    Summary:
    """
    result = llm(prompt, max_tokens=200)
    summaries.append(result['choices'][0]['text'].strip())
    
  final_summary = " ".join(summaries)
  
  os.makedirs("assets/summaries", exist_ok=True)
  output_path = os.path.join("assets/summaries", output_name)
  
  with open(output_path, "w", encoding="utf-8") as f:
    f.write(final_summary)
  
  return final_summary
