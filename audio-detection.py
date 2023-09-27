import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def load_audio_file(file_path):
    audio, sample_rate = librosa.load(file_path)
    return audio, sample_rate

def extract_mfccs(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate)
    return mfccs

def load_dataset(dataset_path):
    dataset = np.load(dataset_path, allow_pickle=True)
    return dataset

def preprocess_dataset(dataset):
    mfccs = []
    labels = []

    for item in dataset:
        mfcc = extract_mfccs(item[0], item[1])
        mfccs.append(mfcc)
        labels.append(item[2])

    mfccs = np.array(mfccs)
    labels = np.array(labels)

    return mfccs, labels

def train_model(mfccs, labels):
    X_train, X_test, y_train, y_test = train_test_split(mfccs, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

def test_model(model, mfccs):
    predictions = model.predict(mfccs)
    return predictions
