"""Genera las matrices de confusión normalizadas, las curvas ROC y Precision-Recall multiclase (08 a 10)
y el diagrama integral de arquitectura del programa y del Perceptrón Multicapa (12)."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from src.paths import DATA, MODELS, RESULTS, FIGURES
from src.plotting_theme import apply_theme, save_figure, CLASS_COLORS, COLORS
from proyecto_utils import TARGET, CLASS_ORDER


def plot_12_architecture():
    """Genera el diagrama de la estructura del programa (flujo de módulos) y del Perceptrón Multicapa ganador."""
    apply_theme()
    fig = plt.figure(figsize=(16, 10.5), facecolor=COLORS["seubg"])
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.26)

    # =========================================================================
    # PANEL A: ARQUITECTURA MODULAR DEL PROGRAMA Y FLUJO DE DATOS SIN FUGA
    # =========================================================================
    ax_prog = fig.add_subplot(gs[0])
    ax_prog.set_xlim(0, 100)
    ax_prog.set_ylim(0, 32)
    ax_prog.axis("off")
    ax_prog.set_title(
        "A. Estructura Modular del Programa y Flujo de Datos (Cero Fuga de Información)",
        fontsize=13,
        fontweight="bold",
        color="#1B2631",
        pad=10,
    )

    stages = [
        (
            2.0,
            7.0,
            17.5,
            20.5,
            COLORS["seuamber"],
            "1. Preprocesamiento\n(preprocesamiento.py)",
            "• Entrada: 1,000,000 filas\n• Rescate MXN (7,562), léxico\n  (22,438) y gasto (30,000)\n• Eliminados: 30k nulos +\n  30k fuera de rango (60,000)\n• Salida: 940,000 registros",
        ),
        (
            22.0,
            7.0,
            17.5,
            20.5,
            COLORS["seuteal"],
            "2. Muestreo de Cochran\n(muestreo.py)",
            "• Confianza 99% (Z=2.576)\n• Error e=0.5% -> n=61,985\n• Estratificación exacta:\n  - train.csv: 43,389 (70%)\n  - validation.csv: 9,298 (15%)\n  - test.csv: 9,298 (15%)",
        ),
        (
            42.0,
            7.0,
            17.5,
            20.5,
            COLORS["lachaiseblue"],
            "3. Validación Cruzada\n(crossVal.py)",
            "• Exclusivo sobre train.csv\n• Evalúa k in {3, 5, 10}\n• Regla: >=500 casos de\n  Alto_valor por pliegue\n• Selección óptima: k* = 5\n  (588 casos Alto_valor/fold)",
        ),
        (
            62.0,
            7.0,
            17.5,
            20.5,
            COLORS["seupurple"],
            "4. Entrenamiento MLP\n(entrenamiento.py)",
            "• Priors: gamma* = 0.50 (1-SE)\n• GridSearchCV: 472 combos\n  (2,360 fits, 0 fallos)\n• Ganador: SGD + Nesterov\n  (mu=0.95), tanh, (128, 64)\n• Época parada temprana: 29",
        ),
        (
            82.0,
            7.0,
            16.5,
            20.5,
            "#2C3E50",
            "5. Calibración y Eval.\n(calibration / large)",
            "• FrozenEstimator (isotonic)\n  y multiplicadores en val\n• Bootstrap Test (B=1,000):\n  F1-Macro = 0.3383\n• Evaluación masiva en lotes:\n  896,611 filas (40,321 filas/s)",
        ),
    ]

    for x, y_box, w, h, color, title, body in stages:
        box = FancyBboxPatch(
            (x, y_box),
            w,
            h,
            boxstyle="round,pad=0.5,rounding_size=1.2",
            linewidth=2.0,
            edgecolor=color,
            facecolor="white",
            zorder=2,
        )
        ax_prog.add_patch(box)
        header = FancyBboxPatch(
            (x, y_box + h - 5.2),
            w,
            5.2,
            boxstyle="round,pad=0.3,rounding_size=1.0",
            linewidth=0,
            facecolor=color,
            zorder=3,
        )
        ax_prog.add_patch(header)
        ax_prog.text(
            x + w / 2,
            y_box + h - 2.6,
            title,
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="white",
            zorder=4,
        )
        ax_prog.text(
            x + 0.8,
            y_box + h - 6.6,
            body,
            ha="left",
            va="top",
            fontsize=8.0,
            color="#1B2631",
            linespacing=1.32,
            zorder=4,
        )

    for idx in range(len(stages) - 1):
        x1 = stages[idx][0] + stages[idx][2] + 0.4
        x2 = stages[idx + 1][0] - 0.4
        ym = stages[idx][1] + stages[idx][3] / 2
        arrow = FancyArrowPatch(
            (x1, ym),
            (x2, ym),
            arrowstyle="-|>,head_length=6,head_width=4",
            linewidth=2.2,
            color="#34495E",
            zorder=5,
        )
        ax_prog.add_patch(arrow)

    ax_prog.text(
        50,
        1.8,
        "Orquestador Principal: main.py   |   Módulos compartidos: proyecto_utils.py, src/feature_engineering.py, src/balancing.py, src/calibration.py, modelos_referencia.py",
        ha="center",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#EAF2F8", edgecolor=COLORS["lachaiseblue"], lw=1.2),
    )

    # =========================================================================
    # PANEL B: ARQUITECTURA INTERNA DEL PERCEPTRÓN MULTICAPA (MLP) GANADOR
    # =========================================================================
    ax_mlp = fig.add_subplot(gs[1])
    ax_mlp.set_xlim(0, 100)
    ax_mlp.set_ylim(0, 56)
    ax_mlp.axis("off")
    ax_mlp.set_title(
        "B. Arquitectura del Perceptrón Multicapa Ganador: Pipeline (55 Entradas -> 128 tanh -> 64 tanh -> 5 Clases Softmax)",
        fontsize=13,
        fontweight="bold",
        color="#1B2631",
        pad=10,
    )

    # Pre-etapa a la izquierda: Transformación de 30 -> 38 -> 55
    pre_box = FancyBboxPatch(
        (1.8, 10.5),
        16.2,
        36.0,
        boxstyle="round,pad=0.5,rounding_size=1.2",
        linewidth=1.8,
        edgecolor=COLORS["seuteal"],
        facecolor="white",
        zorder=2,
    )
    ax_mlp.add_patch(pre_box)
    ax_mlp.text(
        9.9,
        43.5,
        "Ingeniería y Escalado\n(Dentro de cada Fold)",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["seuteal"],
    )
    ax_mlp.text(
        2.6,
        39.5,
        "1) Variables crudas (29):\n"
        "   • 23 numéricas\n"
        "   • 6 categóricas\n"
        "   (id_cliente descartado)\n\n"
        "2) DomainFeatureTransformer:\n"
        "   +8 variables de negocio\n"
        "   • ratio_descuento\n"
        "   • frecuencia_anualizada\n"
        "   • intensidad_recencia\n"
        "   • indice_valor_cliente...\n\n"
        "3) ColumnTransformer:\n"
        "   • StandardScaler (31 num)\n"
        "   • OneHotEncoder (24 bin)\n"
        "   Total = 55 entradas (x)",
        ha="left",
        va="top",
        fontsize=7.8,
        color="#1B2631",
        linespacing=1.26,
    )

    # Coordenadas de las 4 capas de la red neuronal
    layer_x = [31.5, 51.5, 70.5, 88.5]
    input_y = np.linspace(13.0, 43.0, 7)
    h1_y = np.linspace(11.0, 45.0, 8)
    h2_y = np.linspace(14.0, 42.0, 6)
    out_y = np.linspace(13.5, 42.5, 5)

    # Flecha desde preprocesador hacia capa de entrada
    arr_pre = FancyArrowPatch(
        (18.5, 46.8),
        (30.0, 46.8),
        connectionstyle="arc3,rad=-0.12",
        arrowstyle="-|>,head_length=6,head_width=4",
        linewidth=1.8,
        color=COLORS["seuteal"],
        zorder=4,
    )
    ax_mlp.add_patch(arr_pre)

    # Dibujar conexiones sinápticas entre capas
    for y1 in input_y:
        for y2 in h1_y:
            ax_mlp.plot([layer_x[0], layer_x[1]], [y1, y2], color=COLORS["lachaiseblue"], alpha=0.16, lw=0.8, zorder=1)
    for y1 in h1_y:
        for y2 in h2_y:
            ax_mlp.plot([layer_x[1], layer_x[2]], [y1, y2], color=COLORS["seupurple"], alpha=0.18, lw=0.85, zorder=1)
    for y1 in h2_y:
        for y2 in out_y:
            ax_mlp.plot([layer_x[2], layer_x[3]], [y1, y2], color=COLORS["seuamber"], alpha=0.22, lw=1.0, zorder=1)

    # Etiquetas de neuronas de entrada (de arriba hacia abajo: x_1 a x_55)
    input_labels = [
        "x_1: compras_365d",
        "x_2: gasto_total_mxn",
        "x_3: dias_ult_compra",
        "...",
        "x_31: indice_valor",
        "x_32..54: One-Hot",
        "x_55: metodo_pago",
    ]
    for idx, y_pos in enumerate(input_y[::-1]):
        c = Circle((layer_x[0], y_pos), 1.35, facecolor=COLORS["seublue"], edgecolor="#1B4F72", lw=1.4, zorder=4)
        ax_mlp.add_patch(c)
        ax_mlp.text(layer_x[0] - 1.8, y_pos, input_labels[idx], ha="right", va="center", fontsize=7.4, color="#1B2631", fontweight="bold")

    # Neuronas Capa Oculta 1 (128 neuronas, tanh, de arriba hacia abajo)
    h1_labels = ["h1_1", "h1_2", "h1_3", "h1_4", "...", "h1_126", "h1_127", "h1_128"]
    for idx, y_pos in enumerate(h1_y[::-1]):
        c = Circle((layer_x[1], y_pos), 1.45, facecolor=COLORS["seupurple"], edgecolor="#4A235A", lw=1.4, zorder=4)
        ax_mlp.add_patch(c)
        ax_mlp.text(layer_x[1], y_pos, h1_labels[idx], ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=5)

    # Neuronas Capa Oculta 2 (64 neuronas, tanh, de arriba hacia abajo)
    h2_labels = ["h2_1", "h2_2", "h2_3", "...", "h2_63", "h2_64"]
    for idx, y_pos in enumerate(h2_y[::-1]):
        c = Circle((layer_x[2], y_pos), 1.45, facecolor="#7D3C98", edgecolor="#4A235A", lw=1.4, zorder=4)
        ax_mlp.add_patch(c)
        ax_mlp.text(layer_x[2], y_pos, h2_labels[idx], ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=5)

    # Neuronas Capa de Salida (5 clases Softmax)
    out_classes = [
        ("Ocasional", "39.18%", CLASS_COLORS["Ocasional"]),
        ("Inactivo", "32.50%", CLASS_COLORS["Inactivo"]),
        ("Frecuente", "12.39%", CLASS_COLORS["Frecuente"]),
        ("Promocional", "9.15%", CLASS_COLORS["Promocional"]),
        ("Alto_valor", "6.78%", CLASS_COLORS["Alto_valor"]),
    ]
    for idx, y_pos in enumerate(out_y[::-1]):
        cls_name, prop_txt, cls_color = out_classes[idx]
        c = Circle((layer_x[3], y_pos), 1.55, facecolor=cls_color, edgecolor="#1B2631", lw=1.5, zorder=4)
        ax_mlp.add_patch(c)
        ax_mlp.text(layer_x[3], y_pos, f"p_{idx+1}", ha="center", va="center", fontsize=7.0, color="white", fontweight="bold", zorder=5)
        ax_mlp.text(
            layer_x[3] + 2.1,
            y_pos,
            f"{cls_name}\n({prop_txt})",
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=cls_color,
        )

    # Títulos superiores de cada capa y dimensiones de matrices de pesos
    layer_headers = [
        (layer_x[0], "Capa de Entrada\n55 variables\n(Estandarizadas)", COLORS["lachaiseblue"]),
        (layer_x[1], "Capa Oculta 1\n128 neuronas\nh^(1) = tanh(W^(1)x + b^(1))", COLORS["seupurple"]),
        (layer_x[2], "Capa Oculta 2\n64 neuronas\nh^(2) = tanh(W^(2)h^(1) + b^(2))", "#7D3C98"),
        (layer_x[3], "Capa de Salida\n5 neuronas Softmax\np_c = exp(z_c)/sum(exp(z_j))", COLORS["seuamber"]),
    ]
    for lx, ltxt, lcol in layer_headers:
        ax_mlp.text(
            lx,
            50.2,
            ltxt,
            ha="center",
            va="center",
            fontsize=8.6,
            fontweight="bold",
            color=lcol,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=lcol, lw=1.4),
        )

    # Anotaciones de matrices de pesos entrenables entre capas
    weight_annotations = [
        ((layer_x[0] + layer_x[1]) / 2, 8.2, "W^(1): 55 x 128 + 128 bias\n= 7,168 parámetros"),
        ((layer_x[1] + layer_x[2]) / 2, 8.2, "W^(2): 128 x 64 + 64 bias\n= 8,256 parámetros"),
        ((layer_x[2] + layer_x[3]) / 2, 8.2, "W^(3): 64 x 5 + 5 bias\n= 325 parámetros"),
    ]
    for wx, wy, wtxt in weight_annotations:
        ax_mlp.text(
            wx,
            wy,
            wtxt,
            ha="center",
            va="center",
            fontsize=7.8,
            fontweight="bold",
            color="#2C3E50",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7", edgecolor=COLORS["seuamber"], lw=1.0),
        )

    # Barra inferior con resumen de optimización (Momentum + Priors + Total de parámetros)
    ax_mlp.text(
        50,
        2.2,
        "Total de parámetros entrenables: 15,749   |   Optimizador: SGD con Momentum de Nesterov (mu = 0.95, lr = 0.001, batch = 256, L2 alpha = 0.001)\n"
        "Sinergia: Priors amortiguados (gamma* = 0.50: w_Alto_valor = 1.72 vs w_Ocasional = 0.71) corrigen el desbalance y Momentum amortigua la varianza del gradiente",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="bold",
        color="#1B2631",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=COLORS["seupurple"], lw=1.5),
    )

    save_figure(fig, FIGURES / "12_arquitectura_programa_y_perceptron.png")
    plt.close(fig)


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

    plot_12_architecture()


if __name__ == "__main__":
    main()
