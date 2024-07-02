import librosa
import numpy as np
import joblib

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

class EmotionPredictor:
    def __init__(self, model_path='emotion_model.joblib', emotions_path='emotions.joblib'):
        self.model = joblib.load(model_path)
        self.emotions = joblib.load(emotions_path)

    def predict_emotion(self, file_path):
        try:
            # Extract features from the audio file
            features = extract_features(file_path)
            
            # Reshape features to match the input shape expected by the model
            features = features.reshape(1, -1)
            
            # Predict emotion
            predicted = self.model.predict(features)[0]
            
            # Check if the prediction is already a string (emotion name)
            if isinstance(predicted, str):
                emotion = predicted
            else:
                # If it's an index, use it to get the emotion from the list
                emotion = self.emotions[predicted]
            
            return emotion
        except Exception as e:
            print(f"Error predicting emotion: {str(e)}")
            return "unknown"

    def get_available_emotions(self):
        return self.emotions

# Example usage
if __name__ == "__main__":
    predictor = EmotionPredictor()
    
    # Print available emotions
    print("Available emotions:", predictor.get_available_emotions())
    
    # Example prediction
    test_file = "path_to_test_audio_file.wav"  # Replace with a path to a test audio file
    predicted_emotion = predictor.predict_emotion(test_file)
    print(f"Predicted emotion for {test_file}: {predicted_emotion}")