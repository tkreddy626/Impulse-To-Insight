from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import logging
from T import convert_mp3_to_wav, perform_speaker_diarization, split_audio_by_speaker, transcribe_conversation
from Plain import convert_mp3_to_wav as plain_convert_mp3_to_wav, transcribe_audio_to_text, summarize_text

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
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
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['audioFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.endswith('.mp3'):
            wav_file_path = convert_mp3_to_wav(file_path)
        else:
            wav_file_path = file_path

        try:
            flags = perform_speaker_diarization(wav_file_path)
            if not flags:
                return jsonify({'error': 'Speaker diarization failed'})

            segmented_audio_files = split_audio_by_speaker(wav_file_path, flags)
            transcript = transcribe_conversation(segmented_audio_files)

            return jsonify({'transcript': transcript})
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            return jsonify({'error': 'An error occurred during transcription.'}), 500
        finally:
            os.remove(wav_file_path)

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['audioFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.endswith('.mp3'):
            wav_file_path = plain_convert_mp3_to_wav(file_path)
        else:
            wav_file_path = file_path

        try:
            text = transcribe_audio_to_text(wav_file_path)
            summary = summarize_text(text)
            return jsonify({'summary': summary})
        except Exception as e:
            logging.error(f"Error during summarization: {e}")
            return jsonify({'error': 'An error occurred during summarization.'}), 500
        finally:
            os.remove(wav_file_path)

if __name__ == '__main__':
    app.run(debug=True)
