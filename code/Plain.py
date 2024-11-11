import os
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

# Summarizer pipeline
summarizer = pipeline('summarization', model="facebook/bart-large-cnn")

def convert_mp3_to_wav(mp3_file):
    """Convert MP3 to WAV."""
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = mp3_file.replace(".mp3", ".wav")
    audio.export(wav_file, format="wav")
    return wav_file

def transcribe_audio_to_text(wav_file):
    """Transcribe audio to text."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text

def summarize_text(text):
    try:
        input_length = len(text.split())
        max_length = min(0.5 * input_length, 150)
        min_length = min(0.2 * input_length, 50)
        summary = summarizer(text, max_length=int(max_length), min_length=int(min_length), do_sample=False)
        return summary[0]['summary_text'] if summary else ""
    except Exception as e:
        raise RuntimeError(f"Error in text summarization: {str(e)}")

if __name__ == "__main__":
    mp3_file_path = "Call.mp3" 

    wav_file = convert_mp3_to_wav(mp3_file_path)
    text = transcribe_audio_to_text(wav_file)
    summary = summarize_text(text)
    print(summary)
    os.remove(wav_file)
