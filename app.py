from flask import Flask, request, render_template
import os
import tempfile
from werkzeug.utils import secure_filename
from emotion_predictor import EmotionPredictor

app = Flask(__name__)

predictor = EmotionPredictor()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error='No selected file')
        if file:
            # Save the file temporarily
            temp_path = 'temp_audio.wav'
            file.save(temp_path)
            
            # Predict emotion
            emotion = predictor.predict_emotion(temp_path)
            
            # Remove temporary file
            os.remove(temp_path)
            
            result = {
                'emotion': emotion
            }
    
    return render_template('upload.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)