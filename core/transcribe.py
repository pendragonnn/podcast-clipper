import whisper
import os, time

def transcribe_audio(audio_path, output_name="transcript.txt"):
  model = whisper.load_model("base")
  result = model.transcribe(audio_path)
  transcript_text = result["text"]
  
  os.makedirs("assets/transcripts", exist_ok=True)
  output_path = os.path.join("assets/transcripts", output_name)
  
  with open(output_path, "w", encoding="utf-8") as f:
    f.write(transcript_text)
  
  return transcript_text