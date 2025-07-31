from core.download import download_podcast
from core.transcribe import transcribe_audio
from core.text_utils import clean_transcript
from core.summarize import generate_summary_with_llama
from core.voiceover import generate_voiceover
from core.image_gen import generate_image
from core.video import create_vertical_video

if __name__ == "__main__":
  url = "https://www.youtube.com/watch?v=nogh434ykF0"
  audio, video_id, title = download_podcast(url)
  print(f"Downloaded: {title}")
  raw = transcribe_audio(audio, f"{video_id}_transcript.txt")
  cleaned = clean_transcript(raw)
  summary = generate_summary_with_llama(cleaned, f"{video_id}_summary.txt")
  narration = generate_voiceover(summary, f"{video_id}_narration.mp3")
  image = generate_image("Podcast background, cinematic, dramatic lighting, blurred, minimalist", f"{video_id}_bg.jpg")
  video = create_vertical_video(image, narration, summary, f"{video_id}_short.mp4")
  print(f"Created: {video}")
