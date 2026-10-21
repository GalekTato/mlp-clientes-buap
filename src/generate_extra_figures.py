"""Completa matrices test/local y curvas ROC-PR multiclase."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from src.paths import DATA, MODELS, RESULTS, FIGURES
from src.plotting_theme import apply_theme, save_figure, CLASS_COLORS, COLORS
from proyecto_utils import TARGET, CLASS_ORDER

def main():
    apply_theme()
    model_path = MODELS / "mejor_modelo_mlp.joblib"
    if not model_path.exists():
        model_path = MODELS / "mejor_modelo_mlp_v2.joblib"
    model = joblib.load(model_path)
    test = pd.read_csv(DATA / "test.csv")
    y = test.pop(TARGET); pred = model.predict(test)
    result_path = RESULTS / "validacion_masiva.json"
    if not result_path.exists():
        result_path = RESULTS / "validacion_masiva_v2.json"
    large = json.loads(result_path.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y, pred, labels=CLASS_ORDER, normalize="true"), annot=True, fmt=".2f", ax=axes[0], cmap="Purples", xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    sns.heatmap(np.asarray(large["confusion_matrix"]) / np.maximum(np.asarray(large["confusion_matrix"]).sum(axis=1, keepdims=True), 1), annot=True, fmt=".2f", ax=axes[1], cmap="Blues", xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER)
    axes[0].set_title("Test"); axes[1].set_title("Archivo local completo")
    save_figure(fig, FIGURES / "08_matrices_confusion_test_y_archivo.png")
    save_figure(fig, FIGURES / "08_matrices_confusion_test_y_1M.png"); plt.close(fig)
    proba = model.predict_proba(test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, cls in enumerate(model.classes_):
        binary = (y == cls).astype(int)
        fpr, tpr, _ = roc_curve(binary, proba[:, i]); precision, recall, _ = precision_recall_curve(binary, proba[:, i])
        color = CLASS_COLORS.get(cls, COLORS["seugray"])
        axes[0].plot(fpr, tpr, label=f"{cls} ({auc(fpr,tpr):.2f})", color=color)
        axes[1].plot(recall, precision, label=cls, color=color)
    axes[0].plot([0,1], [0,1], "--", color=COLORS["seugray"]); axes[0].set_title("ROC One-vs-Rest"); axes[1].set_title("Precision-Recall")
    for ax in axes: ax.set_xlabel("Tasa"); ax.set_ylabel("Puntuación"); ax.legend(fontsize=8)
    save_figure(fig, FIGURES / "09_curvas_roc_pr_multiclase.png"); plt.close(fig)

if __name__ == "__main__":
    main()
