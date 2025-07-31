# 🎙️ Video Summary Generator

An automated pipeline to download, transcribe, summarize, voiceover, and generate vertical video clips from YouTube Video. The final result is a short-form video ready for TikTok, Reels, or Shorts.

## 🚀 Features
- Download podcast audio from YouTube
- Automatic transcription using Whisper
- Automatic summarization using LLaMA
- Voiceover generation from the summary
- Background image generation using AI prompt
- Final vertical video generation

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourname/podcast-clipper.git
cd podcast-clipper
```

### 2. Set Up a Virtual Environment
```bash
python -m venv podcast-nv
.\podcast-nv\Scripts\activate  # On Windows
# or
source podcast-nv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required Models
LLaMA Model
Download the file llama-2-7b-chat.Q4_K_M.gguf from HuggingFace
Then, place the file in the root directory of this project:

link: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

Whisper model will be automatically handled by the library (via whisper or faster-whisper).

▶️ How to Use
1. Open main.py

2. Modify the YouTube video URL inside:

3. Run the script:
```bash
python main.py
```

## Output Files
🗒️ Transcript: transcripts/{video_id}.txt

🧾 Summary: summaries/{video_id}.txt

🔊 Narration audio: {video_id}_narration.mp3

🖼️ Background image: {video_id}_bg.jpg

🎥 Final vertical video: {video_id}_short.mp4

📁 Folder Structure
``` bash
├── core/
│   ├── download.py         # Download audio from YouTube
│   ├── transcribe.py       # Transcribe audio using Whisper
│   ├── summarize.py        # Summarize using LLaMA
│   ├── voiceover.py        # Generate voiceover
│   ├── image_gen.py        # Generate AI image
│   ├── video.py            # Combine all into a video
│   └── text_utils.py       # Helpers for chunking and cleaning
├── transcripts/            # Folder for raw transcripts
├── summaries/              # Folder for generated summaries
├── main.py                 # Entry point script
├── requirements.txt
└── README.md
```

## ⚡ Performance Tips

1. Use GPU for faster processing:

- Whisper and LLaMA both benefit from CUDA acceleration.

- Ensure you have nvidia-cuda and cudnn installed correctly.

Enjoy generating short podcast videos effortlessly! 🎉
