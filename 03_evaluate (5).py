#Theo Pizzardi
#Alexandre Medor
#Wiam Essadki
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, cohen_kappa_score
from sklearn.preprocessing import label_binarize

os.makedirs("resultats", exist_ok=True)

# Charger le modele et les donnees de test
modele = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")
classes = joblib.load("model/classes.pkl")
X_test = np.load("model/X_test.npy")
y_test = np.load("model/y_test.npy")

print("Evaluation sur", len(X_test), "echantillons")
print("Classes :", classes)

# Predictions
y_pred = modele.predict(X_test)
y_proba = modele.predict_proba(X_test)

# Metriques globales
acc = float(np.mean(y_pred == y_test))
kappa = float(cohen_kappa_score(y_test, y_pred))
auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))

print(f"\nAccuracy : {acc*100:.2f}%")
print(f"AUC-ROC  : {auc:.4f}")
print(f"Kappa    : {kappa:.4f}")
print()
print(classification_report(y_test, y_pred, target_names=classes))

# Sauvegarder le rapport texte
with open("resultats/rapport.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy : {acc*100:.2f}%\n")
    f.write(f"AUC-ROC  : {auc:.4f}\n")
    f.write(f"Kappa    : {kappa:.4f}\n\n")
    f.write(classification_report(y_test, y_pred, target_names=classes))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("AW2A3-IA - Matrice de Confusion", fontweight="bold")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axes[0])
axes[0].set_title("Valeurs brutes")
axes[0].set_ylabel("Reel")
axes[0].set_xlabel("Predit")

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axes[1], vmin=0, vmax=1)
axes[1].set_title("Normalisee")
axes[1].set_ylabel("Reel")
axes[1].set_xlabel("Predit")

plt.tight_layout()
plt.savefig("resultats/confusion_matrix.png", dpi=150)
plt.close()
print("Sauvegarde : resultats/confusion_matrix.png")

# Courbes ROC
y_bin = label_binarize(y_test, classes=range(len(classes)))
colors = ["#B45309", "#C0181A", "#5B21B6", "#0D7F6E", "#0284C7"]

fig, ax = plt.subplots(figsize=(8, 7))
for i, (classe, couleur) in enumerate(zip(classes, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
    auc_i = float(roc_auc_score(y_bin[:, i], y_proba[:, i]))
    ax.plot(fpr, tpr, color=couleur, lw=2, label=f"{classe} (AUC={auc_i:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatoire")
ax.set_xlabel("Taux Faux Positifs")
ax.set_ylabel("Taux Vrais Positifs")
ax.set_title(f"Courbes ROC - AUC macro = {auc:.3f}", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("resultats/roc_curves.png", dpi=150)
plt.close()
print("Sauvegarde : resultats/roc_curves.png")

# Importance des features (si Random Forest)
try:
    importances = modele.feature_importances_
    noms_features = (
        [f"MFCC_{i}" for i in range(40)] +
        [f"dMFCC_{i}" for i in range(40)] +
        [f"ddMFCC_{i}" for i in range(40)] +
        ["ZCR", "RMS", "Centroid", "Rolloff", "Bandwidth"] +
        [f"Chroma_{i}" for i in range(12)]
    )

    top15 = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(15), importances[top15][::-1], color="#0284C7")
    ax.set_yticks(range(15))
    ax.set_yticklabels([noms_features[i] for i in top15][::-1])
    ax.set_title("Top 15 features importantes", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig("resultats/feature_importance.png", dpi=150)
    plt.close()
    print("Sauvegarde : resultats/feature_importance.png")
except:
    print("(importance des features non disponible pour ce modele)")

print("\n--> Lance maintenant : python 04_predict.py data/raw/asthma/fichier.wav")