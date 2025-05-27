# ML_DL_Projet_3
Modèle Deep Learning de type End-to-End -Audio-

###
<h2 align="left">Projet 3 : Audio</h2>

###

<p align="left">Réalisé par :EL MAJDOUBI Adil & OUAHMANE Abdallah</p>
<p align="left">Encadé par : Pr .MAHMOUDI Abdelhak</p>


<h2 align="left">environment setup</h2>

###

<div align="left">
  <img src="https://avatars.githubusercontent.com/u/22800682?s=48&v=4" height="40" alt="javascript logo"  />
  <img width="12" />
  <img src="https://gravatar.com/avatar/5fcb1033abfa7fdb91f995d4035f6544" height="40" alt="typescript logo"  />
  <img width="12" />
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1869px-Python-logo-notext.svg.png" height="40" alt="react logo"  />
  <img width="12" />
  <img src="https://colab.research.google.com/img/colab_favicon_256px.png" height="40" alt="react logo"  />
  <img width="12" />
  <img src="https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-512.png" height="40" alt="react logo"  />
  <img width="12" />

</div>

###




<h2 align="left">Entraining et déploiement le modèle</h2>

Ce script télécharge automatiquement un dataset audio/text depuis Kaggle (sur les versets du Coran), en utilisant la bibliothèque kagglehub, pour un projet de reconnaissance vocale.

```bash
import kagglehub
bigguyubuntu_quran_ayat_speech_to_text_path = kagglehub.dataset_download('bigguyubuntu/quran-ayat-speech-to-text')
print('Data source import complete.')
```

Installer la bibliothèque pydub dans un environnement, il permet de manipuler des fichiers audios de façon simple.

```bash
pip install numpy joblib pydub librosa scikit-learn
pip install numpy pydub librosa tensorflow
pip install pydub
```

Une série d’importations de bibliothèques Python utilisées pour un projet deep learning / machine learning appliqué à de l'audio, probablement pour la reconnaissance vocale.

```bash
import os
import joblib
import torch
import torchaudio
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio.transforms as T
import tensorflow as tf

from tqdm import tqdm
from pydub import AudioSegment
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay
```

Définir une liste de dossiers audio de récitateurs du Coran, accompagnée de noms d'identifiants plus simples.

```bash
folders = [
    ("AbdulSamad_64kbps", "abdulsamad"),
    ("Abdul_Basit_Murattal_64kbps", "abdulbasit"),
    ("Abdullaah_3awwaad_Al-Juhaynee_128kbps", "abdallah_3waad"),
    ("Abdullah_Basfar_32kbps", "abdallah_basfar"),
    ("Abdurrahmaan_As-Sudais_64kbps", "sudais"),
    ("Abu_Bakr_Ash-Shaatree_64kbps", "shatree"),
    ("Alafasy_64kbps", "afasy"),
    ("Ali_Jaber_64kbps", "ali_jaber"),
    ("Ayman_Sowaid_64kbps", "ayman_sowaid"),
    ("Banna_32kbps", "banna"),
    ("Fares_Abbad_64kbps", "fares_abbad"),
    ("Ghamadi_40kbps", "ghamadi"),
    ("Hani_Rifai_192kbps", "hani_rifai"),
    ("Hudhaify_64kbps", "hudhaify"),
    ("Husary_64kbps", "husary"),
    ("Ibrahim_Akhdar_32kbps", "ibrahim_akhdar"),
    ("Maher_AlMuaiqly_64kbps", "maher_almuaiqly"),
    ("Minshawy_Murattal_128kbps", "minshawy"),
    ("Mohammad_al_Tablaway_64kbps", "tablawy"),
    ("Mostafa_Ismail_128kbps", "mostafa_ismail"),
    ("Muhammad_Ayyoub_64kbps", "mohammed_ayyoub"),
    ("Muhammad_Jibreel_64kbps", "mohammed_jibreel"),
    ("Muhsin_Al_Qasim_192kbps", "muhsin_alqasim"),
    ("Saood_ash-Shuraym_64kbps", "soaad"),
    ("Yaser_Salamah_128kbps", "yaser_salamah"),
    ("Yasser_Ad-Dussary_128kbps", "yasser_addussary"),
]
```


Renvoie le nombre d’éléments (tuples) dans la liste folders.
```bash
len(folders)
```

Initialiser deux listes vides


```bash
X = []
y = []
```


Une fonction bien conçue pour extraire les coefficients MFCC à partir d’un fichier audio en utilisant PyTorch + torchaudio.

```bash
def extract_mfcc_features(file_path):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        mfccs = torchaudio.compliance.kaldi.mfcc(waveform=waveform.cuda(), sample_frequency=sample_rate)
        return mfccs.mean(dim=0).cpu().numpy()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
```

Afficher et vérifier le chemin de dataset
```bash
print(f'Downloaded dataset path: {bigguyubuntu_quran_ayat_speech_to_text_path}')
```


Parcourt tous les récitateurs listés dans folders.
Accède à leurs enregistrements audios dans le dossier téléchargé via kagglehub.
Extrait les MFCCs moyens pour chaque fichier.
Les stocke dans X (features) et y (étiquettes du récitant).

```bash
for folder, reader in folders:
    # Construct the full path by including the correct subdirectory structure
    reader_folder_path = os.path.join(bigguyubuntu_quran_ayat_speech_to_text_path, 'Quran_Ayat_public', 'audio_data', folder)

    # Check if the directory exists before listing its contents
    if os.path.exists(reader_folder_path):
        audio_list = os.listdir(reader_folder_path)
        print(f'Processing {folder} ({len(audio_list)} fichiers)')

        for audio in tqdm(audio_list):
            audio_path = os.path.join(reader_folder_path, audio)
            features = extract_mfcc_features(audio_path)

            if features is not None:
                X.append(features)
                y.append(reader)
    else:
        print(f"Warning: Directory not found: {reader_folder_path}")
```


Créer un DataFrame Pandas à partir de la liste y
```bash
print(len(X), len(y))
```

Assemblage de tes données audio et labels dans un seul DataFrame.
```bash
features_df = pd.DataFrame(X)
df = pd.concat([df['Class'], features_df], axis=1)
```

Affiche les 5 premières lignes de ton DataFrame df.
```bash
df.head()
```

Sauvegarder DataFrame df (labels + features MFCC) dans un fichier CSV
```bash
df.to_csv('./quran-readers-audio-processed.csv')
```

LabelEncoder de sklearn.preprocessing sert à transformer des labels non numériques (par exemple des noms comme "sudais", "afasy", "abdulsamad") en nombres entiers (ex: 0, 1, 2…) que les modèles ML peuvent manipuler facilement.

```bash
le = LabelEncoder()
```
fit() : apprend la correspondance entre chaque label texte unique dans y et un entier unique.
transform() : convertit ta liste de labels textuels y en une liste (ou tableau) d’entiers correspondants.
```bash
y_transformed = le.fit_transform(y)
```


C’est un tableau NumPy ou une liste d’entiers qui correspondent à tes labels texte encodés.
Chaque entier représente un récitant unique, selon l’ordre appris par LabelEncoder.
```bash
y_transformed
```

Activer l'exécution "eager" (immédiate) des fonctions TensorFlow.
```bash
tf.config.run_functions_eagerly(True)
```
Implement Gaussian Naive Bayes Classifier using TensorFlow
```bash
class GaussianNaiveBayes(tf.Module):
    def __init__(self):
        super().__init__()
        self.classes = None
        self.mean = None
        self.var = None
        self.prior = None
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32), tf.TensorSpec(shape=[None], dtype=tf.int64)])
    def fit(self, X, y):
        X = X.numpy()
        y = y.numpy()
        self.classes = np.unique(y)
        self.mean = np.zeros((len(self.classes), X.shape[1]), dtype=np.float32)
        self.var = np.zeros((len(self.classes), X.shape[1]), dtype=np.float32)
        self.prior = np.zeros(len(self.classes), dtype=np.float32)
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = np.mean(X_c, axis=0)
            self.var[idx, :] = np.var(X_c, axis=0)
            self.prior[idx] = X_c.shape[0] / float(X.shape[0])
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
    def predict(self, X):
        def _calculate_likelihood(class_idx, X):
            mean = self.mean[class_idx]
            var = self.var[class_idx]
            numerator = tf.exp(-(X - mean) ** 2 / (2 * var))
            denominator = tf.sqrt(2 * np.pi * var)
            return numerator / denominator
        def _calculate_posterior(X):
            posteriors = []
            for idx in range(len(self.classes)):
                prior = tf.math.log(self.prior[idx])
                likelihood = tf.reduce_sum(tf.math.log(_calculate_likelihood(idx, X)), axis=1)
                posterior = prior + likelihood
                posteriors.append(posterior)
            return tf.transpose(tf.convert_to_tensor(posteriors))
        posteriors = _calculate_posterior(X)
        predictions = tf.argmax(posteriors, axis=1)
        return predictions
```

Créer une instance de ton classificateur Naive Bayes
```bash
gnb = GaussianNaiveBayes()
```

Split the data into training and testing sets
```bash
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=42)
gnb.fit(tf.convert_to_tensor(X_train), tf.convert_to_tensor(y_train))
```

Pour éviter les erreurs, il vaut mieux écrire explicitement les types lors de la conversion en tenseurs TensorFlow.
```bash
gnb.fit(tf.convert_to_tensor(X_train, dtype=tf.float32), tf.convert_to_tensor(y_train, dtype=tf.int64))
```
Passer X_test tel quel à gnb.predict, alors que ta méthode attend un tenseur TensorFlow de type tf.float32.
```bash
y_pred = gnb.predict(X_test)
```



y_pred est un tenseur TensorFlow, il faut d’abord le convertir en tableau NumPy pour que accuracy_score, precision_score, et recall_score fonctionnent correctement.
```bash
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {precision * 100:.2f}%')
```

Convertir le modèle TensorFlow en TensorFlow Lite
```bash
converter = tf.lite.TFLiteConverter.from_concrete_functions([gnb.predict.get_concrete_function()])
tflite_model = converter.convert()
```

Convertir le  modèle en fichier gnb_model.tflite
```bash
with open('gnb_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("Model successfully converted to TFLite!")
```

Télécharger le modèle
```bash
import joblib
from google.colab import files
# Sauvegarde du modèle
joblib.dump(gnb, "model_gnb.pkl")
# Sauvegarde du label encoder
joblib.dump(le, "label_encoder.pkl")
files.download("model_gnb.pkl")
files.download("label_encoder.pkl")
files.download("gnb_model.tflite")
```
<h2 align="left">environment setup</h2>

```bash
# -------------------------------------------
# Test d'un fichier audio avec le modèle ML
# -------------------------------------------
from pydub import AudioSegment
import numpy as np
# Charger le fichier test.wav (s'assurer qu'il est mono et en 16kHz)
# Assurez-vous que "1_16.wav" existe dans le répertoire courant ou spécifiez le chemin complet
audio = AudioSegment.from_wav("1_16.wav")
audio = audio.set_channels(1).set_frame_rate(16000)
audio.export("test_converted.wav", format="wav")
# Extraire les caractéristiques (MFCC)
# La fonction extract_mfcc_features doit être définie dans une cellule précédente
features = extract_mfcc_features("test_converted.wav")
features = np.array(features).reshape(1, -1)  # Reshape pour correspondre à l’entrée du modèle
# Prédiction
# Use the correct variable name 'gnb' instead of 'model'
prediction = gnb.predict(features)
# Afficher la prédiction
# If you want the original class name, you'll need to inverse transform the prediction
# Assuming 'le' (LabelEncoder) is available from previous cells
predicted_class_index = prediction[0].numpy() # Convert TensorFlow tensor to numpy
predicted_class_name = le.inverse_transform([predicted_class_index])[0] # Inverse transform the predicted index
print(f"Classe prédite (index) : {predicted_class_index}")
print(f"Classe prédite (nom) : {predicted_class_name}")
```

