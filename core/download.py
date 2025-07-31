import os
import yt_dlp as ydl

def download_podcast(url):
  output_dir = "assets/audio"
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
    audio_path = f"{output_dir}/{info['id']}.mp3"
    return audio_path, info['id'], info['title']
