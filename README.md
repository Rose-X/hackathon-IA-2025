# hackathon-IA-2025
TESSAN × SNOWFLAKE HACKATHON
# AW2A3-IA — Analyse Acoustique Respiratoire

Système de détection de pathologies pulmonaires par analyse de sons.
Détecte 5 classes : **Asthme, BPCO, Bronchique, Pneumonie, Sain**

---

## Structure du projet

Avant de commencer, ton dossier doit ressembler à ça :

```
aw2a3-ia/
│
├── data/
│   └── raw/
│       ├── asthma/          ← fichiers .wav des patients asthmatiques
│       ├── Bronchial/       ← fichiers .wav bronchiques
│       ├── copd/            ← fichiers .wav BPCO
│       ├── healthy/         ← fichiers .wav patients sains
│       └── pneumonia/       ← fichiers .wav pneumonie
│
├── 01_extract_features.py
├── 02_train.py
├── 03_evaluate.py
├── 04_predict.py
└── interface.py
```

> Les dossiers `features/`, `model/` et `resultats/` seront créés automatiquement.

---

## Étape 0 — Télécharger le dataset

Va sur ce lien et télécharge le dataset :

```
https://www.kaggle.com/datasets/vuppalaadithyasairam/lung-sound-dataset
```

Une fois téléchargé, copie les fichiers audio dans les bons dossiers :

```
data/raw/asthma/      → tous les fichiers asthme
data/raw/Bronchial/   → tous les fichiers bronchique
data/raw/copd/        → tous les fichiers BPCO
data/raw/healthy/     → tous les fichiers sains
data/raw/pneumonia/   → tous les fichiers pneumonie
```

---

## Étape 1 — Installer les dépendances

Ouvre un terminal et lance :

```bash
pip install librosa numpy pandas scikit-learn matplotlib seaborn joblib tqdm
```

---

## Étape 2 — Lancer les scripts dans l'ordre

### Important — toujours lancer depuis le bon dossier

```bash
cd C:\Users\TonNom\...\aw2a3-ia
```

---

### Script 1 — Extraire les features audio

```bash
python 01_extract_features.py
```

**Ce que ça fait :**
Lit tous les fichiers audio, transforme chaque son en 137 chiffres (MFCC + features spectrales) et sauvegarde le résultat.

**Ce que tu dois voir :**
```
Classes trouvees : ['Bronchial', 'asthma', 'copd', 'healthy', 'pneumonia']
asthma : 285 fichiers
  asthma: 100%|████████| 285/285
...
Termine ! 1211 fichiers traites
Shape des features : (1211, 137)
```

**Fichiers créés dans `features/` :**
- `X.npy` — les 137 features de chaque fichier
- `y.npy` — le label de chaque fichier (0, 1, 2, 3, 4)
- `classes.npy` — les noms des classes

---

### Script 2 — Entraîner le modèle

```bash
python 02_train.py
```

**Ce que ça fait :**
Teste 3 modèles (Random Forest, SVM, MLP) et garde le meilleur automatiquement.

**Ce que tu dois voir :**
```
Entrainement : Random Forest...
  Train : 100.0%  |  Validation : 89.6%

Entrainement : SVM...
  Train : 100.0%  |  Validation : 85.7%

Entrainement : MLP (reseau de neurones)...
  Train : 97.2%   |  Validation : 84.1%

Meilleur modele : Random Forest (89.6%)
```

**Durée estimée :** 2 à 5 minutes selon ta machine.

**Fichiers créés dans `model/` :**
- `best_model.pkl` — le modèle entraîné
- `scaler.pkl` — la normalisation
- `classes.pkl` — les noms des classes
- `comparaison_modeles.png` — graphique de comparaison

---

### Script 3 — Évaluer les résultats

```bash
python 03_evaluate.py
```

**Ce que ça fait :**
Teste le modèle sur des données qu'il n'a jamais vues et génère les graphiques d'évaluation.

**Ce que tu dois voir :**
```
Accuracy : 89.56%
AUC-ROC  : 0.9412
Kappa    : 0.8691

              precision  recall  f1-score
asthma            0.88    0.91      0.89
Bronchial         0.87    0.85      0.86
...
```

**Fichiers créés dans `resultats/` :**
- `rapport.txt` — accuracy, F1, AUC-ROC
- `confusion_matrix.png` — matrice de confusion
- `roc_curves.png` — courbes ROC par classe
- `feature_importance.png` — features les plus importantes

---

### Script 4 — Tester sur un fichier audio

```bash
python 04_predict.py data/raw/asthma/NOM_DU_FICHIER.wav
```

**Exemple :**
```bash
python 04_predict.py data/raw/asthma/P9AsthmaRS_44.wav
```

**Ce que tu dois voir :**
```
Fichier : P9AsthmaRS_44.wav
Diagnostic : ASTHMA
Confiance  : 87.3%

Probabilites par classe :
  asthma       [##########################----] 87.3% <--
  Bronchial    [####---------------------------]  6.1%
  healthy      [###----------------------------]  4.2%
  copd         [##-----------------------------]  1.8%
  pneumonia    [#------------------------------]  0.6%
```

---

### Interface graphique

```bash
python interface.py
```

Ouvre une fenêtre avec un bouton pour choisir un fichier audio et affiche le diagnostic.

---

## Résumé des commandes

```bash
# 1. Installer les dépendances
pip install librosa numpy pandas scikit-learn matplotlib seaborn joblib tqdm

# 2. Se mettre dans le bon dossier
cd C:\Users\TonNom\...\aw2a3-ia

# 3. Lancer dans l'ordre
python 01_extract_features.py
python 02_train.py
python 03_evaluate.py

# 4. Tester un fichier
python 04_predict.py data/raw/asthma/ton_fichier.wav

# 5. Interface graphique
python interface.py
```

---

## Problèmes fréquents

**"FileNotFoundError: data/raw"**
→ Tu n'es pas dans le bon dossier. Lance `cd C:\...\aw2a3-ia` d'abord.

**"ModuleNotFoundError: No module named 'librosa'"**
→ Lance `pip install librosa`

**"No such file: model/best_model.pkl"**
→ Tu as oublié de lancer `02_train.py` avant `03_evaluate.py` ou `04_predict.py`

**Le script se lance depuis VS Code et plante**
→ Utilise le terminal PowerShell directement, pas le bouton Run de VS Code.

---

## Technologies utilisées

| Outil | Rôle |
|---|---|
| `librosa` | Extraction des features audio (MFCC) |
| `scikit-learn` | Entraînement des modèles (Random Forest, SVM, MLP) |
| `numpy` | Manipulation des données numériques |
| `matplotlib / seaborn` | Génération des graphiques |
| `joblib` | Sauvegarde du modèle |
| `tkinter` | Interface graphique |

---

*AW2A3-IA — TESSAN × Snowflake Hackathon 2025*
