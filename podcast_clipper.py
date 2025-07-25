import yt_dlp as ydl
import os
import whisper
import time
import re
from llama_cpp import Llama
from gtts import gTTS

def download_podcast(url):
  """Download audio from YouTube podcast"""
  output_dir = "audio"
  os.makedirs(output_dir, exist_ok=True) 

  ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': f'{output_dir}/%(id)s.%(ext)s', 
    'postprocessors': [{
      'key': 'FFmpegExtractAudio',
      'preferredcodec': 'mp3',
      'preferredquality': '192'
    }]
  }

  with ydl.YoutubeDL(ydl_opts) as ydl_instance:
    info = ydl_instance.extract_info(url, download=True)
    audio_filename = f"{output_dir}/{info['id']}.mp3"
    return audio_filename, info['id'], info['title']

def transcribe_audio(audio_path, model_size="base"):
  """Transcribe audio to text with Whisper"""
  print(f"Loading Whisper model ({model_size})...")
  start_time = time.time()

  model = whisper.load_model(model_size)

  print(f"Transcribing audio: {os.path.basename(audio_path)}")
  result = model.transcribe(audio_path)

  transcript = result["text"]
  elapsed = time.time() - start_time
  print(f"Transcription completed in {elapsed:.2f} seconds")

  return transcript

def save_transcript(video_id, text):
  """Save transcript file"""
  transcript_dir = "transcripts"
  os.makedirs(transcript_dir, exist_ok=True)

  transcript_file = os.path.join(transcript_dir, f"{video_id}.txt")
  with open(transcript_file, "w", encoding="utf-8") as f:
    f.write(text)

  print(f"Transcript saved: {transcript_file}")
  return transcript_file

def clean_transcript(text):
  """Clean up transcript for better summarization and narration"""
  text = re.sub(r'\b(uh|um|ah|mm|oh|like|you know)\b', '', text, flags=re.IGNORECASE)
  text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)  # remove timestamps
  text = re.sub(r'\b(Speaker \d+|[A-Z ]+:)\b', '', text, flags=re.IGNORECASE)  # remove speaker labels
  text = re.sub(r'[^\w\s.,?!]', '', text)
  text = re.sub(r'\s+', ' ', text).strip()
  text = re.sub(r'[.]{2,}', '.', text)
  return text

def approximate_token_count(text: str) -> int:
  return len(text) // 3

def chunk_text(text: str, max_tokens: int) -> list:
  chunks = []
  current_chunk = ""
  for sentence in re.split(r'(?<=[.!?]) +', text):
    if approximate_token_count(current_chunk + sentence) < max_tokens:
      current_chunk += sentence + " "
    else:
      chunks.append(current_chunk.strip())
      current_chunk = sentence + " "
  if current_chunk:
    chunks.append(current_chunk.strip())
  return chunks

def generate_summary_with_llama(text: str) -> str:
  llm = Llama(model_path="./llama-2-7b-chat.Q4_K_M.gguf")
  model_context_limit = 512
  reserved_for_prompt = 256
  chunk_token_limit = model_context_limit - reserved_for_prompt

  print(f"Chunking text with approx {chunk_token_limit} tokens per chunk...")
  chunks = chunk_text(text, chunk_token_limit)
  summaries = []

  for idx, chunk in enumerate(chunks):
    prompt = f"""
        You are an expert summarizer.
        Summarize the following transcript chunk into a concise summary for create a youtube short video.

        Transcript Chunk:
        {chunk}

        Summary:
        """
    print(f"Generating summary for chunk {idx+1}/{len(chunks)}...")
    print(f"Chunk: {chunk}")
    output = llm(prompt, max_tokens=200)
    summary = output['choices'][0]['text'].strip()
    print(f"Summary Output: {summary}")
    summaries.append(summary)

  combined_summary = " ".join(summaries)
  print("All chunks summarized.")
  return combined_summary

def save_summary(video_id, summary, folder="summaries"):
  os.makedirs(folder, exist_ok=True)
  summary_file = os.path.join(folder, f"{video_id}.txt")
  with open(summary_file, "w", encoding="utf-8") as f:
    f.write(summary)
  print(f"Saved to {summary_file}")
  return summary_file

def generate_voiceover(text, output_file="narration.mp3", lang='en'):
  """
    Generate voiceover from text using gTTS (Google Text-to-Speech).
  """
  if not text.strip():
    raise ValueError("Input text for voiceover is empty.")

  output_dir = "narrations"
  os.makedirs(output_dir, exist_ok=True)
  output_path = os.path.join(output_dir, output_file)

  tts = gTTS(text=text, lang=lang, slow=False)
  tts.save(output_path)

  print(f"Voiceover generated: {output_path}")
  return output_path

if __name__ == "__main__":
  url = "https://www.youtube.com/watch?v=nogh434ykF0&t=84s"
  try:
    audio_file, video_id, title = download_podcast(url)
    print(f"Download: {title}")
    print(f"Audio file: {audio_file}")

    transcript = transcribe_audio(audio_file, model_size="base")
    transcript_file = save_transcript(video_id, transcript)

    cleaned_text = clean_transcript(transcript)

    # (1) Generate summary
    summary = generate_summary_with_llama(cleaned_text)
    print("\nGenerated Summary:")
    print(summary)
    save_summary(video_id, summary, folder="summaries")

    # (2) Generate voiceover narration
    narration_file = generate_voiceover(
      text=summary,
      output_file=f"{video_id}_narration.mp3",
      lang='en'
    )
    print(f"Narration audio saved at: {narration_file}")

  except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
