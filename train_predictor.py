import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def get_emotion_ravdess(filename):
    try:
        parts = filename.split('-')
        emotion_code = int(parts[2])
        emotion_map = {
            1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
            5: 'angry', 6: 'fear', 7: 'disgust', 8: 'ps'
        }
        return emotion_map[emotion_code]
    except (IndexError, ValueError):
        print(f"Warning: Unable to parse emotion from filename: {filename}")
        return 'unknown'

def get_emotion_custom(filename):
    try:
        return filename.split('_')[-1].split('.')[0].lower()
    except IndexError:
        print(f"Warning: Unable to parse emotion from custom filename: {filename}")
        return 'unknown'

def get_emotion_emodb(filename):
    emotion_code = filename[5].upper()
    emotion_map = {
        'W': 'angry', 'L': 'boredom', 'E': 'disgust',
        'A': 'fear', 'F': 'happy', 'T': 'sad', 'N': 'neutral'
    }
    return emotion_map[emotion_code]


def prepare_dataset(data_path):
    features = []
    labels = []
    
    print("Processing RAVDESS training data...")
    training_path = os.path.join(data_path, 'training')
    for actor_folder in tqdm(os.listdir(training_path)):
        actor_path = os.path.join(training_path, actor_folder)
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(actor_path, file)
                    feature = extract_features(file_path)
                    features.append(feature)
                    if file.startswith('03'):  # RAVDESS
                        emotion = get_emotion_ravdess(file)
                    else:  # TESS or other
                        emotion = get_emotion_custom(file)
                    if emotion != 'unknown':
                        labels.append(emotion)
                    else:
                        features.pop()  # Remove the last added feature    
    
    print("Processing EMO-DB data...")
    emodb_path = os.path.join(data_path, 'emodb', 'wav')
    for file in tqdm(os.listdir(emodb_path)):
        if file.endswith('.wav'):
            file_path = os.path.join(emodb_path, file)
            feature = extract_features(file_path)
            features.append(feature)
            emotion = get_emotion_emodb(file)
            labels.append(emotion)
    
    print("Processing custom training data...")
    custom_train_path = os.path.join(data_path, 'train-custom')
    for file in tqdm(os.listdir(custom_train_path)):
        if file.endswith('.wav'):
            file_path = os.path.join(custom_train_path, file)
            feature = extract_features(file_path)
            features.append(feature)
            emotion = get_emotion_custom(file)
            labels.append(emotion)
    
    return np.array(features), np.array(labels)

print("Preparing dataset...")
data_path = 'data'  # Replace with your data folder path
features, labels = prepare_dataset(data_path)

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

print("Training model...")
model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300)
model.fit(X_train, y_train)

print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Saving model...")
joblib.dump(model, 'emotion_model.joblib')
print("Model saved as emotion_model.joblib")

print("Saving list of emotions...")
emotions = sorted(list(set(labels)))
joblib.dump(emotions, 'emotions.joblib')
print("Emotions list saved as emotions.joblib")

print("Process completed!")