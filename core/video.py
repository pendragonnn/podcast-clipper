import os
from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip

def create_vertical_video(image_path, audio_path, text, filename):
  out_dir = "assets/clips"
  os.makedirs(out_dir, exist_ok=True)
  W, H = 1080, 1920
  audio = AudioFileClip(audio_path)
  bg = ImageClip(image_path).resized((W, H)).with_duration(audio.duration)
  subtitle = TextClip(text=text, font="C:/Windows/Fonts/arial.ttf", font_size=60,
                        color="white", size=(W - 100, None), method="caption",
                        text_align="center")
  subtitle = subtitle.with_position(("center", H - 300)).with_duration(audio.duration)
  video = CompositeVideoClip([bg, subtitle]).with_audio(audio)
  out_path = os.path.join(out_dir, filename)
  video.write_videofile(out_path, fps=30, codec="libx264", audio_codec="aac")
  return out_path