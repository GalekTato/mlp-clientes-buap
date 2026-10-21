"""Genera las figuras del sistema con la paleta concurrente."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.paths import ROOT, RESULTS, FIGURES, ensure_layout
from src.plotting_theme import apply_theme, save_figure, CLASS_COLORS, COLORS

def main():
    ensure_layout(); apply_theme()
    dist = pd.read_csv(RESULTS / "distribucion_particiones.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = dist.pivot(index="class", columns="partition", values="proportion")
    pivot.plot(kind="bar", ax=ax, color=[COLORS["seupurple"], COLORS["lachaiseblue"], COLORS["seublue"], COLORS["seuteal"], COLORS["seuamber"]])
    ax.set_ylabel("Proporción"); ax.set_xlabel(""); ax.set_title("Distribución poblacional y particiones estratificadas")
    save_figure(fig, FIGURES / "01_distribucion_clases_cochran.png"); plt.close(fig)

    cv = pd.read_csv(RESULTS / "reporte_crossval.csv")
    fig, ax1 = plt.subplots(figsize=(8, 5)); ax2 = ax1.twinx()
    ax1.plot(cv["k"], cv["minority_min_validation"], "o-", color=COLORS["lachaiseblue"], label="Alto_valor por fold")
    ax1.errorbar(cv["k"], cv["mlp_f1_macro_mean"], yerr=cv["mlp_f1_macro_std"], fmt="D-",
                 color=COLORS["seupurple"], label="F1 macro MLP ± desv.")
    ax2.plot(cv["k"], cv["elapsed_seconds"], "s--", color=COLORS["seuamber"], label="Tiempo")
    ax1.set_xlabel("k"); ax1.set_ylabel("Escala: casos / F1 macro"); ax2.set_ylabel("Segundos")
    ax1.set_title("Compromiso de representatividad y costo computacional")
    save_figure(fig, FIGURES / "02_analisis_k_crossval.png"); plt.close(fig)

    grid_path = RESULTS / "resultados_gridsearch_v2.csv"
    if not grid_path.exists(): grid_path = RESULTS / "resultados_gridsearch.csv"
    grid = pd.read_csv(grid_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    top = grid.nlargest(min(10, len(grid)), "mean_test_f1_macro")
    ax.errorbar(range(len(top)), top["mean_test_f1_macro"], yerr=top["std_test_f1_macro"], fmt="o", color=COLORS["seupurple"])
    ax.set_xlabel("Configuración ordenada"); ax.set_ylabel("F1 macro CV"); ax.set_title("Rejilla paralela e IC aproximado por desviación")
    save_figure(fig, FIGURES / "03_evolucion_rejilla_paralela.png"); plt.close(fig)

    eval_path = RESULTS / "evaluacion_splits.csv"
    if eval_path.exists():
        ev = pd.read_csv(eval_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=ev, x="split", y="f1_macro", ax=ax, palette=[COLORS["seupurple"], COLORS["seuteal"], COLORS["lachaiseblue"]], hue="split", legend=False)
        ax.set_title("Convergencia y generalización"); ax.set_ylabel("F1 macro")
        save_figure(fig, FIGURES / "04_curvas_aprendizaje_convergencia.png"); plt.close(fig)

    balance_path = RESULTS / "balance_comparison.json"
    if not balance_path.exists():
        balance_path = RESULTS / "balance_comparison_v2.json"
    if balance_path.exists():
        rows = json.loads(balance_path.read_text(encoding="utf-8"))
        frame = pd.DataFrame([
            {"scheme": row["scheme"], "metric": metric, "score": row[f"{metric}_mean"]}
            for row in rows for metric in ("f1_macro", "balanced_accuracy", "recall_frecuente", "recall_promocional")
            if f"{metric}_mean" in row
        ])
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(data=frame, x="scheme", y="score", hue="metric", ax=ax,
                    palette=[COLORS["seupurple"], COLORS["lachaiseblue"], COLORS["seuteal"], COLORS["seuamber"]])
        ax.set_title("Comparación de balanceo dentro de cada fold"); ax.set_ylim(0, 1)
        ax.set_ylabel("Media entre folds"); ax.set_xlabel("Esquema")
        save_figure(fig, FIGURES / "05_impacto_balanceo_frecuente_promocional.png"); plt.close(fig)

    cal = RESULTS / "calibracion.json"
    if not cal.exists(): cal = RESULTS / "calibracion_v2.json"
    if cal.exists():
        frame = pd.read_json(cal)
        fig, ax = plt.subplots(figsize=(8, 5)); frame.set_index("method")[["log_loss", "brier_multiclass", "ece"]].plot(kind="bar", ax=ax, color=[COLORS["seupurple"], COLORS["seuteal"], COLORS["seublue"]])
        ax.set_title("Métricas probabilísticas antes/después de calibración"); ax.set_ylabel("Valor")
        save_figure(fig, FIGURES / "06_curvas_calibracion_probabilidades.png"); plt.close(fig)

    var_path = RESULTS / "variacion_estadistica_mejor_modelo.csv"
    if var_path.exists():
        frame = pd.read_csv(var_path)
        fig, ax = plt.subplots(figsize=(8, 5)); ax.errorbar(frame["metric"], frame["mean"], yerr=1.96 * frame["SE"], fmt="o", color=COLORS["seupurple"])
        ax.set_title("Variación estadística e IC 95%"); ax.set_ylabel("Media")
        save_figure(fig, FIGURES / "07_variacion_estadistica_boxplots.png"); plt.close(fig)

if __name__ == "__main__":
    main()
