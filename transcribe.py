import subprocess
import os
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # Importing Processor

# Function to download audio using yt-dlp
def download_audio(video_url):
    print("Downloading audio...")
    subprocess.run(['yt-dlp', '-x', '--audio-format', 'mp3', video_url], check=True)
    print("Download completed!")

# Function to preprocess audio
def preprocess_audio(audio_file):
    # Load audio file
    audio, sr = librosa.load(audio_file, sr=16000)  # Resample to 16 kHz
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    return audio

# Function to extract features (not used in transcribing but kept for future use)
def extract_features(audio):
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)  # Average across time
    return mfccs

# Function to transcribe audio
def transcribe_audio(audio_file):
    audio = preprocess_audio(audio_file)
    
    # Load Wav2Vec2 model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")  # Keep as Processor
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()  # Set model to evaluation mode

    # Convert audio to tensor
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000, clean_up_tokenization_spaces=True).input_values

    # Make predictions
    with torch.no_grad():
        logits = model(input_values).logits

    # Get predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the predicted IDs to text
    transcription = processor.decode(predicted_ids[0])

    return transcription

def main():
    video_url = input("Enter YouTube video URL: ")
    download_audio(video_url)
    
    audio_file = 'audio.mp3'  # Make sure the filename matches what yt-dlp saves
    print("Transcribing audio...")
    try:
        transcript = transcribe_audio(audio_file)
        print("Transcript:", transcript)
    except Exception as e:
        print("Transcription failed:", e)

if __name__ == '__main__':
    main()
