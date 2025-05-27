import numpy as np
import tensorflow as tf
import librosa
from pydub import AudioSegment
import joblib

# === Prétraitement de l'audio ===
audio = AudioSegment.from_wav("c:/Users/adilma/Desktop/Projet3_vf/1_16.wav")
audio = audio.set_channels(1).set_frame_rate(16000)
audio.export("c:/Users/adilma/Desktop/Projet3_vf/test_converted.wav", format="wav")

# === Fonction pour extraire les MFCC ===
def extract_mfcc_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Extraire les caractéristiques (et réduire à une seule valeur)
mfcc_features = extract_mfcc_features("c:/Users/adilma/Desktop/Projet3_vf/test_converted.wav")
mean_feature = np.mean(mfcc_features)  # moyenne des 13 MFCCs
features = np.array([[mean_feature]], dtype=np.float32)  # [1, 1]

# === Charger le modèle TensorFlow Lite ===
interpreter = tf.lite.Interpreter(model_path="c:/Users/adilma/Desktop/Projet3_vf/gnb_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Shape attendue :", input_details[0]['shape'])

# Donner les données en entrée
interpreter.set_tensor(input_details[0]['index'], features)

# Lancer la prédiction
interpreter.invoke()

# Récupérer la prédiction
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_index = np.argmax(output_data)

# === Charger le LabelEncoder pour convertir l’index en nom ===
le = joblib.load("c:/Users/adilma/Desktop/Projet3_vf/label_encoder.pkl")

predicted_class = le.inverse_transform([predicted_index])[0]

# === Affichage ===
print(f"Classe prédite (index) : {predicted_index}")
print(f"Classe prédite (nom) : {predicted_class}")
