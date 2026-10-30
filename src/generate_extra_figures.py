"""Genera las matrices de confusión normalizadas y las curvas ROC y Precision-Recall multiclase (08 a 10)."""
from __future__ import annotations
import json
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
    model = joblib.load(model_path)
    test = pd.read_csv(DATA / "test.csv")
    y = test.pop(TARGET)
    proba = model.predict_proba(test)
    classes = np.asarray(model.classes_)
    pred = classes[np.argmax(proba, axis=1)]

    result_path = RESULTS / "validacion_masiva.json"
    large = json.loads(result_path.read_text(encoding="utf-8"))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sns.heatmap(
        confusion_matrix(y, pred, labels=CLASS_ORDER, normalize="true"),
        annot=True,
        fmt=".2f",
        ax=axes[0],
        cmap="Purples",
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
    )
    large_cm = np.asarray(large["confusion_matrix"], dtype=float)
    large_norm = large_cm / np.maximum(large_cm.sum(axis=1, keepdims=True), 1.0)
    sns.heatmap(
        large_norm,
        annot=True,
        fmt=".2f",
        ax=axes[1],
        cmap="Blues",
        xticklabels=CLASS_ORDER,
        yticklabels=CLASS_ORDER,
    )
    axes[0].set_title("Matriz de confusión normalizada en Test (9,298 filas)")
    axes[1].set_title("Evaluación masiva fuera de entrenamiento (896,611 filas)")
    for ax in axes:
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
    fig.tight_layout()
    save_figure(fig, FIGURES / "08_matrices_confusion_test_y_archivo.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    auc_rows = []
    for i, cls in enumerate(classes):
        binary = (y == cls).astype(int)
        fpr, tpr, _ = roc_curve(binary, proba[:, i])
        precision, recall, _ = precision_recall_curve(binary, proba[:, i])
        roc_val = auc(fpr, tpr)
        pr_val = auc(recall, precision)
        auc_rows.append({"class": str(cls), "roc_auc": float(roc_val), "pr_auc": float(pr_val)})
        color = CLASS_COLORS.get(str(cls), COLORS["seugray"])
        axes[0].plot(fpr, tpr, label=f"{cls} (AUC={roc_val:.3f})", color=color, linewidth=2)
        axes[1].plot(recall, precision, label=f"{cls} (PR={pr_val:.3f})", color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], "--", color=COLORS["seugray"])
    axes[0].set_title("Curvas ROC One-vs-Rest en Test")
    axes[0].set_xlabel("Tasa de Falsos Positivos (FPR)")
    axes[0].set_ylabel("Tasa de Verdaderos Positivos (TPR)")
    axes[0].legend(fontsize=8.5, loc="lower right")

    axes[1].set_title("Curvas Precision-Recall por clase en Test")
    axes[1].set_xlabel("Sensibilidad (Recall)")
    axes[1].set_ylabel("Precisión (Precision)")
    axes[1].legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    save_figure(fig, FIGURES / "09_curvas_roc_pr_multiclase.png")
    plt.close(fig)

    auc_frame = pd.DataFrame(auc_rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(auc_frame))
    width = 0.36
    ax.bar(
        x - width / 2,
        auc_frame["roc_auc"],
        width,
        label="ROC-AUC (OvR)",
        color=[CLASS_COLORS.get(c, COLORS["seugray"]) for c in auc_frame["class"]],
    )
    ax.bar(
        x + width / 2,
        auc_frame["pr_auc"],
        width,
        label="PR-AUC",
        color=COLORS["seugray"],
        alpha=0.82,
    )
    ax.set_xticks(x, auc_frame["class"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Área bajo la curva")
    ax.set_title("Desempeño discriminativo por clase en Test: ROC-AUC y PR-AUC")
    ax.legend()
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    save_figure(fig, FIGURES / "10_auc_area_multiclase.png")
    plt.close(fig)
    auc_frame.to_csv(RESULTS / "auc_por_clase.csv", index=False)


if __name__ == "__main__":
    main()
