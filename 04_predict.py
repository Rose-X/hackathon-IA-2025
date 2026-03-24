import sys
import os
import numpy as np
import librosa
import joblib

# Usage : python 04_predict.py data/raw/asthma/fichier.wav
if len(sys.argv) < 2:
    print("Usage : python 04_predict.py chemin/vers/fichier.wav")
    sys.exit()

fichier = sys.argv[1]

if not os.path.exists(fichier):
    print("Fichier introuvable :", fichier)
    sys.exit()

# Charger le modele
modele = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
classes = joblib.load("model/classes.pkl")

# Extraire les features (meme chose que dans 01_extract_features.py)
audio, sr = librosa.load(fichier, sr=22050, duration=3.0, mono=True)
if len(audio) < 22050 * 3:
    audio = np.pad(audio, (0, 22050 * 3 - len(audio)))

mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
mfcc_delta = librosa.feature.delta(mfcc)
mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
rms = np.mean(librosa.feature.rms(y=audio))
centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)

features = np.concatenate([
    np.mean(mfcc, axis=1),
    np.mean(mfcc_delta, axis=1),
    np.mean(mfcc_delta2, axis=1),
    [zcr, rms, centroid, rolloff, bandwidth],
    chroma
]).reshape(1, -1)

features = scaler.transform(features)

# Prediction
probas = modele.predict_proba(features)[0]
idx = np.argmax(probas)
resultat = classes[idx]
confiance = probas[idx] * 100

# Affichage
print(f"\nFichier : {os.path.basename(fichier)}")
print(f"Diagnostic : {resultat.upper()}")
print(f"Confiance  : {confiance:.1f}%")
print()
print("Probabilites par classe :")
for i in np.argsort(probas)[::-1]:
    barre = "#" * int(probas[i] * 30) + "-" * (30 - int(probas[i] * 30))
    fleche = " <--" if i == idx else ""
    print(f"  {classes[i]:<12} [{barre}] {probas[i]*100:.1f}%{fleche}")