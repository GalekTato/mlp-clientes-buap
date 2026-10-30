"""Genera el catálogo de gráficas diagnósticas (01 a 07) con estilo visual unificado."""
from __future__ import annotations
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.paths import MODELS, RESULTS, FIGURES, ensure_layout
from src.plotting_theme import apply_theme, save_figure, COLORS


def plot_01_distribution():
    dist = pd.read_csv(RESULTS / "distribucion_particiones.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = dist.pivot(index="class", columns="partition", values="proportion")
    ordered_cols = [c for c in ["population_clean", "sample_cochran", "train", "validation", "test"] if c in pivot.columns]
    pivot = pivot[ordered_cols]
    pivot.plot(
        kind="bar",
        ax=ax,
        color=[
            COLORS["seupurple"],
            COLORS["lachaiseblue"],
            COLORS["seublue"],
            COLORS["seuteal"],
            COLORS["seuamber"],
        ],
    )
    ax.set_ylabel("Proporción de registros")
    ax.set_xlabel("Clase de cliente")
    ax.set_title("Conservación de proporciones entre población limpia y particiones estratificadas")
    plt.xticks(rotation=15)
    save_figure(fig, FIGURES / "01_distribucion_clases_cochran.png")
    plt.close(fig)


def plot_02_crossval_k():
    cv = pd.read_csv(RESULTS / "reporte_crossval.csv")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(
        cv["k"],
        cv["mlp_f1_macro_mean"],
        "D-",
        color=COLORS["seupurple"],
        linewidth=2.2,
        label="F1-Macro MLP (media CV)",
    )
    ax1.fill_between(
        cv["k"],
        cv["mlp_f1_macro_mean"] - cv["mlp_f1_macro_std"],
        cv["mlp_f1_macro_mean"] + cv["mlp_f1_macro_std"],
        color=COLORS["seupurple"],
        alpha=0.18,
    )
    l2 = ax2.plot(
        cv["k"],
        cv["minority_min_validation"],
        "o--",
        color=COLORS["lachaiseblue"],
        linewidth=2.0,
        label="Casos Alto_valor por fold",
    )
    l3 = ax2.axhline(500, color=COLORS["seuamber"], linestyle=":", linewidth=1.6, label="Umbral mínimo (500 casos)")

    ax1.set_xlabel("Número de pliegues (k)")
    ax1.set_ylabel("F1-Macro en validación cruzada", color=COLORS["seupurple"])
    ax2.set_ylabel("Casos de la clase minoritaria (Alto_valor) por fold", color=COLORS["lachaiseblue"])
    ax1.set_xticks(cv["k"].tolist())
    ax1.set_title("Selección de k = 5: estabilidad de F1-Macro y soporte mínimo por pliegue")

    lines = l1 + l2 + [l3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right")
    save_figure(fig, FIGURES / "02_analisis_k_crossval.png")
    plt.close(fig)


def plot_03_gridsearch_top():
    grid_path = RESULTS / "resultados_gridsearch.csv"
    grid = pd.read_csv(grid_path).dropna(subset=["mean_test_f1_macro"])
    if "param_model__max_iter" in grid.columns:
        grid = grid[grid["param_model__max_iter"] == grid["param_model__max_iter"].min()]
    top = grid.sort_values(
        ["mean_test_f1_macro", "std_test_f1_macro", "mean_fit_time"],
        ascending=[False, True, True],
    ).head(10).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 5))
    x = np.arange(len(top))
    ax.errorbar(
        x,
        top["mean_test_f1_macro"],
        yerr=top["std_test_f1_macro"],
        fmt="o-",
        color=COLORS["seupurple"],
        ecolor=COLORS["lachaiseblue"],
        capsize=4,
        linewidth=1.8,
    )
    labels = [
        f"#{i + 1}\n{row['param_model__solver']} {row['param_model__activation']}\n{row['param_model__hidden_layer_sizes']}"
        for i, row in top.iterrows()
    ]
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_xlabel("Mejores configuraciones evaluadas en la rejilla (472 combinaciones)")
    ax.set_ylabel("F1-Macro CV (± 1 DE)")
    ax.set_title("Desempeño de las 10 mejores configuraciones de la rejilla de hiperparámetros")
    save_figure(fig, FIGURES / "03_evolucion_rejilla_paralela.png")
    plt.close(fig)


def plot_04_convergence_and_splits():
    model_path = MODELS / "mejor_modelo_mlp.joblib"
    eval_path = RESULTS / "evaluacion_splits.csv"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if model_path.exists():
        pipeline = joblib.load(model_path)
        mlp = pipeline.named_steps.get("model") or pipeline.named_steps.get("mlp")
        loss_curve = getattr(mlp, "loss_curve_", [])
        val_scores = getattr(mlp, "validation_scores_", [])
        epochs = np.arange(1, len(loss_curve) + 1)
        ax_loss = axes[0]
        ax_loss.plot(epochs, loss_curve, "o-", color=COLORS["seupurple"], markersize=4, label="Pérdida de entrenamiento (log-loss)")
        ax_loss.axvline(len(loss_curve), color=COLORS["seuamber"], linestyle="--", label=f"Parada temprana (época {len(loss_curve)})")
        ax_loss.set_xlabel("Época")
        ax_loss.set_ylabel("Pérdida (Loss)", color=COLORS["seupurple"])
        ax_loss.set_title("Curva de convergencia por época del MLP ganador")
        if val_scores:
            ax_val = ax_loss.twinx()
            ax_val.plot(
                np.arange(1, len(val_scores) + 1),
                val_scores,
                "s--",
                color=COLORS["seuteal"],
                markersize=3.5,
                alpha=0.85,
                label="Exactitud interna de validación",
            )
            ax_val.set_ylabel("Exactitud validación interna", color=COLORS["seuteal"])
        ax_loss.legend(loc="upper right", fontsize=8)

    if eval_path.exists():
        ev = pd.read_csv(eval_path)
        ax_bar = axes[1]
        x = np.arange(len(ev))
        width = 0.36
        ax_bar.bar(x - width / 2, ev["f1_macro"], width, label="F1-Macro", color=COLORS["seupurple"])
        ax_bar.bar(x + width / 2, ev["balanced_accuracy"], width, label="Balanced Accuracy", color=COLORS["seuteal"])
        ax_bar.set_xticks(x, ev["split"], rotation=18, fontsize=8.5)
        ax_bar.set_ylabel("Puntuación")
        ax_bar.set_ylim(0, 0.50)
        ax_bar.set_title("Generalización en Train, Validation y Test vs. Dummy")
        ax_bar.legend(loc="upper right", fontsize=8.5)
        for container in ax_bar.containers:
            ax_bar.bar_label(container, fmt="%.3f", padding=2, fontsize=7.5)

    fig.tight_layout()
    save_figure(fig, FIGURES / "04_curvas_aprendizaje_convergencia.png")
    plt.close(fig)


def plot_05_gamma_sweep():
    balance_path = RESULTS / "balance_comparison.json"
    if not balance_path.exists():
        return
    rows = json.loads(balance_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame([
        {"scheme": row["scheme"], "metric": metric, "score": row[f"{metric}_mean"]}
        for row in rows
        for metric in ("f1_macro", "balanced_accuracy", "recall_frecuente", "recall_promocional")
        if f"{metric}_mean" in row
    ])
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=frame,
        x="scheme",
        y="score",
        hue="metric",
        ax=ax,
        palette=[COLORS["seupurple"], COLORS["lachaiseblue"], COLORS["seuteal"], COLORS["seuamber"]],
    )
    ax.set_title("Barrido del exponente de ponderación por clase (γ) en validación cruzada")
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Media en 5-fold CV")
    ax.set_xlabel("Esquema de ponderación inversamente proporcional a la frecuencia (γ)")
    save_figure(fig, FIGURES / "05_impacto_balanceo_frecuente_promocional.png")
    plt.close(fig)


def plot_06_calibration():
    cal = RESULTS / "calibracion.json"
    if not cal.exists():
        return
    frame = pd.read_json(cal)
    fig, ax = plt.subplots(figsize=(9, 5))
    frame.set_index("method")[["log_loss", "brier_multiclass", "ece"]].plot(
        kind="bar",
        ax=ax,
        color=[COLORS["seupurple"], COLORS["seuteal"], COLORS["seublue"]],
    )
    ax.set_title("Métricas probabilísticas en Test: base vs. calibración Sigmoid e Isotonic")
    ax.set_ylabel("Valor (menor es mejor)")
    ax.set_xlabel("Método")
    plt.xticks(rotation=0)
    save_figure(fig, FIGURES / "06_curvas_calibracion_probabilidades.png")
    plt.close(fig)


def plot_07_statistical_variation():
    grid_path = RESULTS / "resultados_gridsearch.csv"
    var_boot_path = RESULTS / "variacion_estadistica_test_bootstrap.csv"
    if not grid_path.exists():
        return

    grid_df = pd.read_csv(grid_path)
    best_row = grid_df.sort_values("rank_test_f1_macro").iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    metrics = ["f1_macro", "balanced_accuracy", "accuracy", "roc_auc_ovr"]
    long_rows = []
    for metric in metrics:
        for fold_idx in range(5):
            col = f"split{fold_idx}_test_{metric}"
            if col in best_row:
                long_rows.append({"metric": metric, "score": float(best_row[col])})
    long_df = pd.DataFrame(long_rows)

    sns.boxplot(
        data=long_df,
        x="metric",
        y="score",
        ax=axes[0],
        color=COLORS["seublue"],
        width=0.45,
    )
    sns.stripplot(
        data=long_df,
        x="metric",
        y="score",
        ax=axes[0],
        color=COLORS["seupurple"],
        size=7,
        jitter=False,
    )
    axes[0].set_title("Dispersión entre los 5 pliegues de validación cruzada")
    axes[0].set_xlabel("Métrica")
    axes[0].set_ylabel("Valor en cada fold")
    axes[0].tick_params(axis="x", rotation=18)

    if var_boot_path.exists():
        boot_df = pd.read_csv(var_boot_path)
        x = np.arange(len(boot_df))
        yerr_low = boot_df["mean"] - boot_df["ci95_low"]
        yerr_high = boot_df["ci95_high"] - boot_df["mean"]
        axes[1].errorbar(
            x,
            boot_df["mean"],
            yerr=[yerr_low, yerr_high],
            fmt="D",
            color=COLORS["seupurple"],
            ecolor=COLORS["seuamber"],
            elinewidth=2,
            capsize=5,
            markersize=7,
            label="Media ± IC 95% (1,000 réplicas bootstrap)",
        )
        axes[1].set_xticks(x, boot_df["metric"], rotation=18)
        axes[1].set_title("Intervalos de confianza al 95% en Test (Bootstrap B=1,000)")
        axes[1].set_xlabel("Métrica")
        axes[1].set_ylabel("Estimación en Test")
        axes[1].legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    save_figure(fig, FIGURES / "07_variacion_estadistica_boxplots.png")
    plt.close(fig)


def main():
    ensure_layout()
    apply_theme()
    plot_01_distribution()
    plot_02_crossval_k()
    plot_03_gridsearch_top()
    plot_04_convergence_and_splits()
    plot_05_gamma_sweep()
    plot_06_calibration()
    plot_07_statistical_variation()


if __name__ == "__main__":
    main()
