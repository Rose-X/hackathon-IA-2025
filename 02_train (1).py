#Theo Pizzardi
#Alexandre Medor
#Wiam Essadki

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# ── CONFIG ──────────────────────────────────────
FEATURES_DIR = "features"
MODEL_DIR    = "model"
# ────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 50)
print("  AW2A3-IA -- Entrainement du modele")
print("=" * 50)

# ── CHARGEMENT ──────────────────────────────────
print("\n[1/4] Chargement des features...")
X       = np.load(os.path.join(FEATURES_DIR, "X.npy"))
y       = np.load(os.path.join(FEATURES_DIR, "y.npy"))
CLASSES = np.load(os.path.join(FEATURES_DIR, "classes.npy"))

print(f"   X shape   : {X.shape}")
print(f"   y shape   : {y.shape}")
print(f"   Classes   : {list(CLASSES)}")

# ── SPLIT ───────────────────────────────────────
print("\n[2/4] Split train / val / test (70/15/15)...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)
print(f"   Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)
np.save(os.path.join(MODEL_DIR, "X_train.npy"), X_train)
np.save(os.path.join(MODEL_DIR, "y_train.npy"), y_train)

# ── ENTRAINEMENT 3 MODELES ───────────────────────
print("\n[3/4] Entrainement de 3 modeles...")

modeles = {
    "MLP (reseau de neurones)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            verbose=False
        ))
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=10,
            gamma="scale",
            probability=True,
            random_state=42
        ))
    ]),
}

resultats = {}
meilleur_nom = None
meilleure_acc = 0

for nom, pipeline in modeles.items():
    print(f"\n   --> {nom}...")
    pipeline.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, pipeline.predict(X_train))
    acc_val   = accuracy_score(y_val,   pipeline.predict(X_val))
    print(f"       Train : {acc_train*100:.1f}%  |  Val : {acc_val*100:.1f}%")
    resultats[nom] = {"pipeline": pipeline, "val_acc": acc_val, "train_acc": acc_train}
    if acc_val > meilleure_acc:
        meilleure_acc = acc_val
        meilleur_nom  = nom

# ── SAUVEGARDE ──────────────────────────────────
print(f"\n[4/4] Meilleur modele : {meilleur_nom} ({meilleure_acc*100:.1f}%)")
best_pipeline = resultats[meilleur_nom]["pipeline"]
joblib.dump(best_pipeline, os.path.join(MODEL_DIR, "best_model.pkl"))
joblib.dump(CLASSES,       os.path.join(MODEL_DIR, "classes.pkl"))
joblib.dump(best_pipeline.named_steps["scaler"], os.path.join(MODEL_DIR, "scaler.pkl"))

# ── GRAPHIQUE ───────────────────────────────────
noms   = list(resultats.keys())
accs_t = [resultats[n]["train_acc"]*100 for n in noms]
accs_v = [resultats[n]["val_acc"]*100   for n in noms]
x      = np.arange(len(noms))
w      = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - w/2, accs_t, w, label="Train",      color="#0284C7", alpha=0.85)
b2 = ax.bar(x + w/2, accs_v, w, label="Validation", color="#0D7F6E", alpha=0.85)
ax.set_ylabel("Accuracy (%)"); ax.set_title("AW2A3-IA -- Comparaison modeles", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(noms); ax.set_ylim([0, 105])
ax.legend(); ax.grid(axis="y", alpha=0.3)
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f"{b.get_height():.1f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "comparaison_modeles.png"), dpi=150, bbox_inches="tight")
plt.close()

print(f"\n{'='*50}")
print(f"  Termine ! Meilleur : {meilleur_nom}")
print(f"  Val accuracy : {meilleure_acc*100:.2f}%")
print(f"  Modele sauvegarde : {MODEL_DIR}/best_model.pkl")
print(f"\n  --> Lance maintenant : python 03_evaluate.py")
print(f"{'='*50}")
