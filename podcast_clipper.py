import yt_dlp as ydl
import os
import whisper
import time

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
  """transcript audio to text with Whisper"""
  print(f"Loading Whisper model ({model_size})...")
  start_time = time.time()

  # Load model Whisper
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

if __name__ == "__main__":
  url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  try:
    # Download podcast
    audio_file, video_id, title = download_podcast(url)
    print(f"Download: {title}")
    print(f"Audio file: {audio_file}")

    # Transcribe audio
    transcript = transcribe_audio(audio_file, model_size="base")

    # Save transcript
    transcript_file = save_transcript(video_id, transcript)

    # Print first 200 characters for verification
    print("\nTranscript snippet:")
    print(transcript[:200] + "....")

  except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()