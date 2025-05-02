<h1 align="center">🗣️🎧 YouTube Audio Transcriber with Wav2Vec2</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python">
  <img src="https://img.shields.io/badge/Wav2Vec2-HuggingFace-red?logo=github">
  <img src="https://img.shields.io/badge/yt--dlp-Audio_Downloader-yellow?logo=youtube">
</p>

<p align="center">
  🔊 Automatically download audio from YouTube videos and convert speech to text using a deep learning model!
</p>

---

## 🌟 Overview

This project allows you to input any YouTube video URL, download its audio using `yt-dlp`, and generate a transcript using the `facebook/wav2vec2-base-960h` model from Hugging Face. Perfect for speech-to-text tasks, lecture transcription, or YouTube captioning.

---

## 🎯 Features

✅ Download and extract audio directly from YouTube  
✅ Preprocess audio for transcription  
✅ Use Wav2Vec2 transformer for speech-to-text  
✅ Simple CLI-based workflow  
✅ Plug-and-play script with no UI overhead  

---

## 📂 File Structure

youtube-transcriber/
├── transcriber.py # Main script
├── requirements.txt # Python dependencies
└── README.md # This file

yaml
Copy
Edit

---

## 🧑‍💻 How to Run on Your System

### 🛠️ Step 1: Clone the Repo

```bash
git clone https://github.com/yourusername/youtube-transcriber.git
cd youtube-transcriber
🐍 Step 2: Set Up Virtual Environment (Recommended)
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate  # For Windows
source venv/bin/activate  # For macOS/Linux
📦 Step 3: Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
📥 Step 4: Install yt-dlp
bash
Copy
Edit
pip install yt-dlp
🔧 If yt-dlp is not found, you may need to install it via pipx or manually from yt-dlp GitHub.

▶️ Step 5: Run the Script
bash
Copy
Edit
python transcriber.py
Then paste any YouTube URL when prompted, like:

less
Copy
Edit
Enter YouTube video URL: https://www.youtube.com/watch?v=dQw4w9WgXcQ
The audio will be downloaded and transcribed. Output will be printed to the terminal.

🧠 Model Used
facebook/wav2vec2-base-960h
A pretrained model from Hugging Face, designed for automatic speech recognition (ASR) trained on 960 hours of Librispeech data.

🧾 Example Output
less
Copy
Edit
Enter YouTube video URL: https://www.youtube.com/watch?v=example
Downloading audio...
Download completed!
Transcribing audio...
Transcript: Never gonna give you up never gonna let you down never gonna run around and desert you
⚠️ Notes
Ensure yt-dlp is installed and in your PATH.

The audio is saved as audio.mp3. If the output filename differs, modify the audio_file variable.

Performance depends on the audio quality and speaker clarity.

📦 requirements.txt
txt
Copy
Edit
torch
transformers
librosa
numpy
yt-dlp
