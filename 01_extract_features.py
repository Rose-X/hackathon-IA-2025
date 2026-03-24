#Theo Pizzardi
#Alexandre Medor
#Wiam Essadki
import os
import numpy as np
import librosa
from tqdm import tqdm

# Dossier avec les fichiers audio classés par sous-dossier
DATA_DIR = "data/raw"
os.makedirs("features", exist_ok=True)

# On detecte les classes automatiquement depuis les noms de dossiers
classes = sorted(os.listdir(DATA_DIR))
print("Classes trouvees :", classes)

X = []
y = []

for i, classe in enumerate(classes):
    dossier = os.path.join(DATA_DIR, classe)
    fichiers = [f for f in os.listdir(dossier) if f.endswith((".wav", ".mp3"))]
    print(f"{classe} : {len(fichiers)} fichiers")

    for fichier in tqdm(fichiers, desc=classe):
        chemin = os.path.join(dossier, fichier)

        try:
            # Charger le fichier audio, 3 secondes max
            audio, sr = librosa.load(chemin, sr=22050, duration=3.0, mono=True)

            # Padding si le fichier est trop court
            if len(audio) < 22050 * 3:
                audio = np.pad(audio, (0, 22050 * 3 - len(audio)))

            # Extraire les MFCC (40 coefficients)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            # Quelques features spectrales en plus
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            rms = np.mean(librosa.feature.rms(y=audio))
            centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
            chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)

            # On prend la moyenne de chaque coefficient sur le temps
            features = np.concatenate([
                np.mean(mfcc, axis=1),
                np.mean(mfcc_delta, axis=1),
                np.mean(mfcc_delta2, axis=1),
                [zcr, rms, centroid, rolloff, bandwidth],
                chroma
            ])

            X.append(features)
            y.append(i)

        except Exception as e:
            print(f"  Erreur sur {fichier} : {e}")

# Sauvegarder
X = np.array(X)
y = np.array(y)
np.save("features/X.npy", X)
np.save("features/y.npy", y)
np.save("features/classes.npy", np.array(classes))

print(f"\nTermine ! {len(X)} fichiers traites")
print(f"Shape des features : {X.shape}")
