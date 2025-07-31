import os
from gtts import gTTS

def generate_voiceover(text, filename):
  out_dir = "assets/narrations"
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, filename)
  tts = gTTS(text=text, lang='en')
  tts.save(out_path)
  return out_path