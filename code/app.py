from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from functions import convert_mp3_to_wav, perform_speaker_diarization, split_audio_by_speaker, transcribe_conversation

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['audioFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        if filename.endswith('.mp3'):
            wav_file_path = convert_mp3_to_wav(file_path)
        else:
            wav_file_path = file_path
        
        flags = perform_speaker_diarization(wav_file_path)
        if not flags:
            return jsonify({'error': 'Speaker diarization failed'})
        
        segmented_audio_files = split_audio_by_speaker(wav_file_path, flags)
        transcript = transcribe_conversation(segmented_audio_files)
        
        return jsonify({'transcript': transcript})

if __name__ == '__main__':
    app.run(debug=True)
