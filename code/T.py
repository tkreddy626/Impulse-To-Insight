import speech_recognition as sr
from pyAudioAnalysis import audioSegmentation
from pydub import AudioSegment
import warnings
import numpy as np
import io
import wave

warnings.filterwarnings("ignore")

def convert_mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = mp3_file.replace(".mp3", ".wav")
    audio.export(wav_file, format="wav")
    return wav_file

def perform_speaker_diarization(audio_file, num_speakers=2):
    flags = audioSegmentation.speaker_diarization(audio_file, n_speakers=num_speakers)
    return flags[0].tolist() if flags else []

def split_audio_by_speaker(audio_file, flags):
    audio = AudioSegment.from_wav(audio_file)
    segmented_audio_files = []
    start = 0
    current_speaker = flags[0]
    for i, flag in enumerate(flags[1:], start=1):
        if flag != current_speaker:
            end = i
            start_time = start * 0.2 * 1000  
            end_time = end * 0.2 * 1000  
            segment = audio[start_time:end_time]
            segmented_audio_files.append((start, end, segment, current_speaker))
            start = i
            current_speaker = flag
    
    end = len(flags)
    start_time = start * 0.2 * 1000  
    end_time = end * 0.2 * 1000  
    segment = audio[start_time:end_time]
    segmented_audio_files.append((start, end, segment, current_speaker))
    return segmented_audio_files

def transcribe_conversation(audio_files):
    conversation_transcript = ""
    for i, audio_info in enumerate(audio_files):
        start, end, audio_segment, speaker = audio_info
        person = f"Person {speaker + 1}"  
        text = audio_to_text(audio_segment)
        if text:
            conversation_transcript += f"{person}: {text}.\n"
    return conversation_transcript

def audio_to_text(audio_segment):
    try:
        recognizer = sr.Recognizer()
        with io.BytesIO() as wav_buffer:
            audio_segment.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
            audio_data = sr.AudioFile(wav_buffer)
            with audio_data as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
        return text.strip() if text else ""
    except sr.RequestError:
        return ""
    except sr.UnknownValueError:
        return ""
    except Exception:
        return ""
