import yt_dlp as ydl
import os

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
  
if __name__ == "__main__":
  url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  try:
    audio_file, video_id, title = download_podcast(url)
    print(f"Download: {title}")
    print(f"Audio file: {audio_file}")
  except Exception as e:
    print(f"Error: {e}")