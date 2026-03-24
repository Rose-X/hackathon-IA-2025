                             #IA -- Interface  

#Theo Pizzardi
#Alexandre Medor
#Wiam Essadki

# -*- coding: utf-8 -*-
"""
AW2A3-IA -- Interface Medicale
Lance : python interface.py
"""

import sys, os, threading, tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────
MODEL_DIR  = "model"
SR         = 22050
DURATION   = 3.0
N_MFCC     = 40
N_FFT      = 2048
HOP_LENGTH = 512

# Palette clinique — blanc, gris chaud, bleu médical
C = {
    "bg":          "#F4F7FA",
    "surface":     "#FFFFFF",
    "panel":       "#EEF3F8",
    "blue":        "#1565A8",
    "blue_light":  "#2A7DD4",
    "blue_xlight": "#E8F1FB",
    "teal":        "#0B7A6E",
    "teal_light":  "#E6F5F3",
    "border":      "#C8D8E8",
    "border_light":"#DDE8F2",
    "text":        "#1A2B3C",
    "text_mid":    "#3D5266",
    "muted":       "#6B8299",
    "muted_light": "#9AAFC0",
    "white":       "#FFFFFF",
    # classes
    "asthma":      "#B45309",
    "asthma_bg":   "#FFFBEB",
    "copd":        "#9B1C1C",
    "copd_bg":     "#FFF5F5",
    "Bronchial":   "#5B21B6",
    "Bronchial_bg":"#F5F3FF",
    "healthy":     "#065F46",
    "healthy_bg":  "#ECFDF5",
    "pneumonia":   "#9A3412",
    "pneumonia_bg":"#FFF7ED",
}

LABELS_FR = {
    "asthma":    "Asthme",
    "Bronchial": "Bronchique",
    "copd":      "BPCO",
    "healthy":   "Sain",
    "pneumonia": "Pneumonie",
}

ICONS = {
    "asthma":    "◎",
    "Bronchial": "◉",
    "copd":      "◈",
    "healthy":   "✓",
    "pneumonia": "◆",
}

RECOMMANDATIONS = {
    "asthma":    ("Consultation recommandée",
                  "Des signes compatibles avec un asthme bronchique ont été détectés.\n"
                  "Une consultation avec un pneumologue est conseillée dans les 48 heures\n"
                  "pour confirmer le diagnostic et adapter le traitement."),
    "copd":      ("Bilan respiratoire recommandé",
                  "Des signes évocateurs d'une BPCO ont été identifiés.\n"
                  "Un bilan spirométrique complet est recommandé. Votre médecin\n"
                  "pourra évaluer la nécessité d'une prise en charge adaptée."),
    "Bronchial": ("Suivi médical conseillé",
                  "Des anomalies bronchiques ont été détectées dans l'enregistrement.\n"
                  "Une consultation médicale est conseillée pour évaluer\n"
                  "la nécessité d'un traitement bronchodilatateur."),
    "healthy":   ("Résultat rassurant",
                  "Aucune anomalie respiratoire significative n'a été détectée.\n"
                  "Vos poumons semblent fonctionner normalement. Continuez\n"
                  "à prendre soin de votre santé respiratoire."),
    "pneumonia": ("Consultation médicale urgente",
                  "Des signes compatibles avec une pneumonie ont été détectés.\n"
                  "Une consultation médicale rapide est fortement recommandée.\n"
                  "Ce résultat doit être confirmé par un professionnel de santé."),
}


def extract_features(file_path):
    import librosa
    y, _ = librosa.load(file_path, sr=SR, duration=DURATION, mono=True)
    target = int(SR * DURATION)
    y = np.pad(y, (0, max(0, target - len(y))), mode="constant")[:target]
    mfcc   = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.concatenate([
        np.mean(mfcc, axis=1), np.mean(delta, axis=1), np.mean(delta2, axis=1),
        [np.mean(librosa.feature.zero_crossing_rate(y)),
         np.mean(librosa.feature.rms(y=y)),
         np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)),
         np.mean(librosa.feature.spectral_rolloff(y=y, sr=SR)),
         np.mean(librosa.feature.spectral_bandwidth(y=y, sr=SR))],
        np.mean(librosa.feature.chroma_stft(y=y, sr=SR), axis=1),
    ])


class MedicalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AW2A3-IA  —  Analyse Acoustique Respiratoire")
        self.root.geometry("780x700")
        self.root.resizable(False, False)
        self.root.configure(bg=C["bg"])

        # Centrer
        self.root.update_idletasks()
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"780x700+{(sw-780)//2}+{(sh-700)//2}")

        self.pipeline  = None
        self.classes   = None
        self.file_path = None
        self._load_model()
        self._build()

    # ── CHARGEMENT MODELE ──────────────────────
    def _load_model(self):
        try:
            self.pipeline = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
            self.classes  = [str(c) for c in joblib.load(os.path.join(MODEL_DIR, "classes.pkl"))]
        except Exception:
            self.pipeline = None

    # ── CONSTRUCTION UI ────────────────────────
    def _build(self):
        self._build_header()
        self._build_patient_area()
        self._build_upload_area()
        self._build_btn()
        self._build_results_area()
        self._build_footer()

    def _build_header(self):
        h = tk.Frame(self.root, bg=C["blue"], height=64)
        h.pack(fill="x")
        h.pack_propagate(False)

        # Logo + nom
        left = tk.Frame(h, bg=C["blue"])
        left.pack(side="left", padx=20, pady=12)
        tk.Label(left, text="AW2A3-IA",
                 font=("Georgia", 18, "bold"),
                 bg=C["blue"], fg=C["white"]).pack(side="left")
        tk.Label(left, text="  Analyse Acoustique Respiratoire",
                 font=("Georgia", 10),
                 bg=C["blue"], fg="#A8C8E8").pack(side="left", pady=2)

        # Statut connexion
        right = tk.Frame(h, bg=C["blue"])
        right.pack(side="right", padx=20)

        dot_col = "#4ADE80" if self.pipeline else "#F87171"
        dot     = tk.Canvas(right, width=10, height=10, bg=C["blue"], highlightthickness=0)
        dot.pack(side="left", pady=2)
        dot.create_oval(1, 1, 9, 9, fill=dot_col, outline="")
        status = "Système opérationnel" if self.pipeline else "Modèle non chargé"
        tk.Label(right, text=status, font=("Calibri", 9),
                 bg=C["blue"], fg="#A8C8E8").pack(side="left", padx=6)

        # Ligne colorée sous le header
        tk.Frame(self.root, bg="#1176C8", height=3).pack(fill="x")
        tk.Frame(self.root, bg=C["border_light"], height=1).pack(fill="x")

    def _build_patient_area(self):
        frame = tk.Frame(self.root, bg=C["bg"], pady=14)
        frame.pack(fill="x", padx=24)

        # Titre section
        tk.Label(frame, text="DOSSIER PATIENT",
                 font=("Calibri", 8, "bold"),
                 bg=C["bg"], fg=C["muted"]).pack(anchor="w")

        card = tk.Frame(frame, bg=C["surface"],
                        highlightbackground=C["border"],
                        highlightthickness=1)
        card.pack(fill="x", pady=(6, 0))

        inner = tk.Frame(card, bg=C["surface"], pady=12, padx=16)
        inner.pack(fill="x")

        # Avatar
        av = tk.Canvas(inner, width=44, height=44, bg=C["blue_xlight"],
                       highlightbackground=C["border"], highlightthickness=1)
        av.pack(side="left")
        av.create_text(22, 22, text="🧑", font=("Segoe UI Emoji", 18))

        # Info patient
        info = tk.Frame(inner, bg=C["surface"])
        info.pack(side="left", padx=14)
        tk.Label(info, text="Patient en cours d'analyse",
                 font=("Calibri", 12, "bold"),
                 bg=C["surface"], fg=C["text"]).pack(anchor="w")
        tk.Label(info, text="Cabine TESSAN  ·  FR-042  ·  Paris",
                 font=("Calibri", 9),
                 bg=C["surface"], fg=C["muted"]).pack(anchor="w")

        # Badge modele
        badge = tk.Frame(inner, bg=C["blue_xlight"],
                         highlightbackground=C["border"],
                         highlightthickness=1)
        badge.pack(side="right")
        tk.Label(badge, text="Random Forest  ·  89.6% accuracy",
                 font=("Calibri", 9),
                 bg=C["blue_xlight"], fg=C["blue"],
                 padx=10, pady=6).pack()

    def _build_upload_area(self):
        frame = tk.Frame(self.root, bg=C["bg"], pady=4)
        frame.pack(fill="x", padx=24)

        tk.Label(frame, text="ENREGISTREMENT AUDIO",
                 font=("Calibri", 8, "bold"),
                 bg=C["bg"], fg=C["muted"]).pack(anchor="w")

        # Zone de dépôt
        self.drop_frame = tk.Frame(frame,
                                   bg=C["surface"],
                                   highlightbackground=C["border"],
                                   highlightthickness=1,
                                   cursor="hand2")
        self.drop_frame.pack(fill="x", pady=(6, 0))
        self.drop_frame.bind("<Button-1>", lambda e: self._browse())
        self.drop_frame.bind("<Enter>", lambda e: self.drop_frame.config(
            highlightbackground=C["blue_light"]))
        self.drop_frame.bind("<Leave>", lambda e: self.drop_frame.config(
            highlightbackground=C["border"]))

        inner = tk.Frame(self.drop_frame, bg=C["surface"], pady=18)
        inner.pack(fill="x")
        inner.bind("<Button-1>", lambda e: self._browse())

        self.drop_icon = tk.Label(inner, text="♪",
                                  font=("Georgia", 28),
                                  bg=C["surface"], fg=C["muted_light"],
                                  cursor="hand2")
        self.drop_icon.pack()
        self.drop_icon.bind("<Button-1>", lambda e: self._browse())

        self.drop_text = tk.Label(inner,
                                  text="Cliquez pour sélectionner un fichier audio",
                                  font=("Calibri", 11),
                                  bg=C["surface"], fg=C["muted"],
                                  cursor="hand2")
        self.drop_text.pack(pady=(4, 2))
        self.drop_text.bind("<Button-1>", lambda e: self._browse())

        self.drop_sub = tk.Label(inner, text=".wav  ·  .mp3  ·  .flac",
                                 font=("Calibri", 9),
                                 bg=C["surface"], fg=C["muted_light"],
                                 cursor="hand2")
        self.drop_sub.pack()
        self.drop_sub.bind("<Button-1>", lambda e: self._browse())

    def _build_btn(self):
        frame = tk.Frame(self.root, bg=C["bg"], pady=14)
        frame.pack(fill="x", padx=24)

        self.btn = tk.Button(frame,
                             text="Lancer l'analyse acoustique",
                             font=("Calibri", 13, "bold"),
                             bg=C["blue"], fg=C["white"],
                             activebackground="#0F4F8A",
                             activeforeground=C["white"],
                             relief="flat", cursor="hand2",
                             pady=14,
                             command=self._run)
        self.btn.pack(fill="x")

    def _build_results_area(self):
        self.results_frame = tk.Frame(self.root, bg=C["bg"])
        self.results_frame.pack(fill="both", expand=True, padx=24)
        self._show_waiting()

    def _build_footer(self):
        tk.Frame(self.root, bg=C["border_light"], height=1).pack(fill="x")
        foot = tk.Frame(self.root, bg=C["panel"], pady=8)
        foot.pack(fill="x")
        tk.Label(foot,
                 text="AW2A3-IA  ·  TESSAN × Snowflake Hackathon 2025  ·  Ce résultat ne remplace pas un avis médical",
                 font=("Calibri", 8),
                 bg=C["panel"], fg=C["muted_light"]).pack()

    # ── ETATS ─────────────────────────────────
    def _clear_results(self):
        for w in self.results_frame.winfo_children():
            w.destroy()

    def _show_waiting(self):
        self._clear_results()
        tk.Label(self.results_frame,
                 text="Sélectionnez un fichier audio pour démarrer l'analyse",
                 font=("Calibri", 10),
                 bg=C["bg"], fg=C["muted_light"]).pack(expand=True)

    def _show_loading(self):
        self._clear_results()
        frame = tk.Frame(self.results_frame, bg=C["bg"])
        frame.pack(expand=True)
        tk.Label(frame, text="Analyse en cours...",
                 font=("Calibri", 12, "bold"),
                 bg=C["bg"], fg=C["blue"]).pack(pady=(0, 8))
        tk.Label(frame, text="Extraction des features MFCC  ·  Inférence Random Forest",
                 font=("Calibri", 9),
                 bg=C["bg"], fg=C["muted"]).pack()

        # Barre de progression indéterminée
        self.progress = ttk.Progressbar(frame, mode="indeterminate", length=300)
        self.progress.pack(pady=12)
        self.progress.start(12)

    def _show_error(self, msg):
        self._clear_results()
        card = tk.Frame(self.results_frame, bg=C["surface"],
                        highlightbackground="#FCA5A5", highlightthickness=1)
        card.pack(fill="x", pady=8)
        tk.Label(card, text=f"Erreur : {msg}",
                 font=("Calibri", 10),
                 bg=C["surface"], fg=C["copd"],
                 wraplength=680, pady=16).pack()

    # ── ACTIONS ───────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="Sélectionner un fichier audio",
            filetypes=[("Fichiers audio", "*.wav *.mp3 *.flac *.ogg"), ("Tous", "*.*")]
        )
        if path:
            self.file_path = path
            name = os.path.basename(path)
            self.drop_icon.config(text="♫", fg=C["blue"])
            self.drop_text.config(text=name, fg=C["text"],
                                  font=("Calibri", 11, "bold"))
            self.drop_sub.config(text="Fichier prêt pour l'analyse")
            self._show_waiting()

    def _run(self):
        if not self.file_path:
            self._show_error("Veuillez d'abord sélectionner un fichier audio.")
            return
        if not self.pipeline:
            self._show_error("Modèle non chargé. Lancez 02_train.py d'abord.")
            return
        self.btn.config(state="disabled", text="Analyse en cours...")
        self._show_loading()
        threading.Thread(target=self._analyze, daemon=True).start()

    def _analyze(self):
        try:
            feats = extract_features(self.file_path).reshape(1, -1)
            proba = self.pipeline.predict_proba(feats)[0]
            idx   = int(np.argmax(proba))
            cls   = self.classes[idx]
            conf  = float(proba[idx]) * 100
            self.root.after(0, lambda: self._show_results(cls, conf, proba))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
        finally:
            self.root.after(0, lambda: self.btn.config(
                state="normal", text="Lancer l'analyse acoustique"))

    # ── RÉSULTATS ─────────────────────────────
    def _show_results(self, cls, conf, proba):
        self._clear_results()

        cls_col    = C.get(cls, C["blue"])
        cls_bg     = C.get(f"{cls}_bg", C["blue_xlight"])
        nom_fr     = LABELS_FR.get(cls, cls)
        is_healthy = (cls == "healthy")
        reco_title, reco_body = RECOMMANDATIONS.get(cls, ("", ""))

        # ── CARTE RÉSULTAT PRINCIPAL ──────────
        main_card = tk.Frame(self.results_frame, bg=C["surface"],
                             highlightbackground=cls_col, highlightthickness=2)
        main_card.pack(fill="x", pady=(0, 10))

        # Barre colorée en haut
        tk.Frame(main_card, bg=cls_col, height=5).pack(fill="x")

        inner = tk.Frame(main_card, bg=C["surface"], pady=16, padx=20)
        inner.pack(fill="x")

        # Colonne gauche — diagnostic
        left = tk.Frame(inner, bg=C["surface"])
        left.pack(side="left")

        tk.Label(left, text="DIAGNOSTIC AW2A3-IA",
                 font=("Calibri", 8, "bold"),
                 bg=C["surface"], fg=C["muted"]).pack(anchor="w")

        tk.Label(left, text=nom_fr,
                 font=("Georgia", 26, "bold"),
                 bg=C["surface"], fg=cls_col).pack(anchor="w", pady=(2, 4))

        # Chip de confiance
        chip = tk.Frame(left, bg=cls_bg,
                        highlightbackground=cls_col, highlightthickness=1)
        chip.pack(anchor="w")
        tk.Label(chip, text=f"  Confiance : {conf:.1f}%  ",
                 font=("Calibri", 10, "bold"),
                 bg=cls_bg, fg=cls_col,
                 pady=4).pack()

        # Colonne droite — icone
        right = tk.Frame(inner, bg=C["surface"])
        right.pack(side="right")

        icon_canvas = tk.Canvas(right, width=60, height=60,
                                bg=cls_bg, highlightthickness=1,
                                highlightbackground=cls_col)
        icon_canvas.pack()
        icon_text = "✓" if is_healthy else ICONS.get(cls, "◆")
        icon_canvas.create_text(30, 30, text=icon_text,
                                font=("Georgia", 28, "bold"),
                                fill=cls_col)

        # ── BARRES DE PROBABILITÉ ─────────────
        bars_card = tk.Frame(self.results_frame, bg=C["surface"],
                             highlightbackground=C["border"],
                             highlightthickness=1)
        bars_card.pack(fill="x", pady=(0, 10))

        tk.Frame(bars_card, bg=cls_col, height=3).pack(fill="x")

        bars_inner = tk.Frame(bars_card, bg=C["surface"], padx=20, pady=14)
        bars_inner.pack(fill="x")

        tk.Label(bars_inner, text="DISTRIBUTION DES PROBABILITÉS",
                 font=("Calibri", 8, "bold"),
                 bg=C["surface"], fg=C["muted"]).pack(anchor="w", pady=(0, 10))

        sorted_idx = np.argsort(proba)[::-1]
        BAR_W = 460
        for i in sorted_idx:
            c_cls  = self.classes[i]
            p      = float(proba[i])
            c_col  = C.get(c_cls, C["blue"])
            c_bg   = C.get(f"{c_cls}_bg", C["blue_xlight"])
            nom    = LABELS_FR.get(c_cls, c_cls)
            is_top = (i == int(np.argmax(proba)))

            row = tk.Frame(bars_inner, bg=C["surface"])
            row.pack(fill="x", pady=3)

            # Nom
            tk.Label(row, text=nom,
                     font=("Calibri", 10, "bold" if is_top else "normal"),
                     bg=C["surface"],
                     fg=c_col if is_top else C["text_mid"],
                     width=13, anchor="w").pack(side="left")

            # Track
            track = tk.Frame(row, bg=C["border_light"], height=18, width=BAR_W)
            track.pack(side="left", padx=8)
            track.pack_propagate(False)

            fill_w = max(2, int(p * BAR_W))
            fill = tk.Frame(track, bg=c_col if is_top else C["border"],
                            height=18, width=fill_w)
            fill.place(x=0, y=0)

            # Pourcentage
            tk.Label(row, text=f"{p*100:5.1f}%",
                     font=("Calibri", 10, "bold" if is_top else "normal"),
                     bg=C["surface"],
                     fg=c_col if is_top else C["muted"]).pack(side="left", padx=6)

        # ── RECOMMANDATION ────────────────────
        reco_col = C["healthy"] if is_healthy else C["blue"]
        reco_bg  = C["healthy_bg"] if is_healthy else C["blue_xlight"]
        reco_brd = "#6EE7D4" if is_healthy else C["border"]

        reco_card = tk.Frame(self.results_frame, bg=reco_bg,
                             highlightbackground=reco_brd,
                             highlightthickness=1)
        reco_card.pack(fill="x")

        reco_inner = tk.Frame(reco_card, bg=reco_bg, padx=20, pady=14)
        reco_inner.pack(fill="x")

        # Ligne gauche colorée
        tk.Frame(reco_card, bg=reco_col, width=4, height=80).place(x=0, y=0)

        tk.Label(reco_inner, text=reco_title,
                 font=("Calibri", 11, "bold"),
                 bg=reco_bg, fg=reco_col).pack(anchor="w")
        tk.Label(reco_inner, text=reco_body,
                 font=("Calibri", 10),
                 bg=reco_bg, fg=C["text_mid"],
                 justify="left").pack(anchor="w", pady=(6, 0))

        tk.Label(reco_inner,
                 text="Ce résultat est fourni à titre indicatif et ne remplace pas un avis médical professionnel.",
                 font=("Calibri", 8, "italic"),
                 bg=reco_bg, fg=C["muted_light"]).pack(anchor="w", pady=(10, 0))


# ── LANCEMENT ───────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except Exception:
        pass
    app = MedicalApp(root)
    root.mainloop()
