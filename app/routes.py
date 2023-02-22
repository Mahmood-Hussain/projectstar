from app import app
import os
from flask import request, render_template, send_from_directory
from app.models import run_model_on_file

ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/audio', methods=['GET'])
def audio():
    return render_template('audio.html')

@app.route('/audio/<path:filename>')
def send_audio(filename):
    directory = os.path.join(ABS_PATH, os.getenv('CONVERTED_DIR'))
    return send_from_directory(directory, filename)

@app.route('/audio/predict', methods=['POST'])
def audio_predict():
    # upload audio file to server and save it then pass it to the model
    file = request.files['file']
    # create folder to save the file
    os.makedirs(os.getenv('UPLOADS_DIR'), exist_ok=True)
    # save the file
    file_path = os.path.join(os.getenv('UPLOADS_DIR'), file.filename)
    file.save(file_path)
    # pass the file to the model
    result = run_model_on_file(file_path)
    # returned is the path send it to template
    return render_template('converted.html', result=result)
    

