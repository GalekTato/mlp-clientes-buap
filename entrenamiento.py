"""Entrenamiento sin fuga de información de un clasificador MLP multiclase y análisis estadístico."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import joblib
from joblib import Memory, Parallel, delayed, parallel_backend
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.inspection import permutation_importance

from proyecto_utils import (
    CATEGORICAL_COLUMNS,
    CLASS_ORDER,
    NUMERIC_COLUMNS,
    SEED,
    TARGET,
    build_base_pipeline,
    ensure_dir,
    evaluate_predictions,
    make_preprocessor,
    read_dataset,
    write_json,
)
from crossVal import get_stratified_cv
from src.feature_engineering import DomainFeatureTransformer
from src.balancing import sample_weights, oversample
from src.calibration import apply_multipliers, calibrate, probabilistic_metrics, tune_multipliers
from src.plotting_theme import apply_theme, save_figure, COLORS
from src.paths import MODELS, RESULTS, FIGURES, ensure_layout, first_existing


def build_grid(fast: bool = False, max_iter_values: tuple[int, ...] = (150, 300)):
    """Construye el pipeline y el espacio de búsqueda para Adam y SGD con momentum/Nesterov."""
    pipe = build_base_pipeline()
    if fast:
        grid = {
            "model__hidden_layer_sizes": [(64,)],
            "model__max_iter": [80],
            "model__learning_rate_init": [0.001],
            "model__solver": ["adam"],
            "model__activation": ["relu"],
            "model__alpha": [0.001],
            "model__batch_size": [256],
        }
    else:
        grid = [
            {
                "model__hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32)],
                "model__max_iter": list(max_iter_values),
                "model__learning_rate_init": [0.001, 0.005, 0.01],
                "model__solver": ["adam"],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [1e-4, 1e-3, 1e-2],
                "model__batch_size": [256, 512],
            },
            {
                "model__hidden_layer_sizes": [(64,), (128, 64)],
                "model__max_iter": list(max_iter_values),
                "model__learning_rate_init": [0.001, 0.005],
                "model__learning_rate": ["constant", "adaptive"],
                "model__solver": ["sgd"],
                "model__momentum": [0.9, 0.95],
                "model__nesterovs_momentum": [True],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [1e-4, 1e-3],
                "model__batch_size": [256, 512],
            },
        ]
    return pipe, grid


def count_grid_combinations(grid) -> int:
    return len(ParameterGrid(grid))


def scoring():
    return {
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "f1_weighted": make_scorer(f1_score, average="weighted", zero_division=0),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "accuracy": make_scorer(accuracy_score),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "roc_auc_ovr": "roc_auc_ovr",
    }


def evaluate(model, df: pd.DataFrame, split: str):
    """Evalúa un estimador sobre un DataFrame en una sola pasada y devuelve métricas y predicciones."""
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    pred = model.predict(X)
    proba = model.predict_proba(X)
    resumen = evaluate_predictions(y, pred, proba, model.classes_, split)
    return resumen, y, pred, proba


def sweep_gamma(
    X: pd.DataFrame,
    y: pd.Series,
    gammas=(0.0, 0.25, 0.5, 0.75),
    k: int = 5,
    random_state: int = SEED,
    report_dir: str | Path = RESULTS,
    graph_dir: str | Path = FIGURES,
    n_jobs: int = -1,
) -> tuple[float, pd.DataFrame]:
    """Evalúa el exponente de ponderación γ mediante CV estratificada sobre TRAIN y elige γ* por regla 1-SE."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    pipeline_base = build_base_pipeline(max_iter=150, random_state=random_state)
    particiones_cv = list(cv.split(X, y))

    def _evaluar_par(gamma_val: float, fold_idx: int, train_idx, val_idx):
        X_fold, y_fold = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        estimador = clone(pipeline_base)
        argumentos_fit = {}
        if gamma_val > 0.0:
            argumentos_fit["model__sample_weight"] = sample_weights(y_fold, gamma=gamma_val)
        with threadpool_limits(limits=1):
            estimador.fit(X_fold, y_fold, **argumentos_fit)
        pred = estimador.predict(X_val)
        proba = estimador.predict_proba(X_val)
        return {
            "gamma": float(gamma_val),
            "fold": fold_idx,
            "accuracy": float(accuracy_score(y_val, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
            "f1_macro": float(f1_score(y_val, pred, average="macro", zero_division=0)),
            "roc_auc_ovr": float(roc_auc_score(y_val, proba, multi_class="ovr", labels=estimador.classes_)),
        }

    tareas = [
        (gamma_val, fold_idx, train_idx, val_idx)
        for gamma_val in gammas
        for fold_idx, (train_idx, val_idx) in enumerate(particiones_cv, start=1)
    ]
    resultados_folds = Parallel(n_jobs=n_jobs)(
        delayed(_evaluar_par)(g_val, f_idx, t_idx, v_idx)
        for g_val, f_idx, t_idx, v_idx in tareas
    )
    tabla_folds = pd.DataFrame(resultados_folds)

    filas_resumen = []
    for gamma_val in gammas:
        sub = tabla_folds[tabla_folds["gamma"] == float(gamma_val)]
        fila = {"gamma": float(gamma_val)}
        for metrica in ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr"):
            fila[f"{metrica}_mean"] = float(sub[metrica].mean())
            fila[f"{metrica}_std"] = float(sub[metrica].std(ddof=1))
            fila[f"{metrica}_se"] = float(sub[metrica].std(ddof=1) / np.sqrt(k))
        filas_resumen.append(fila)

    df_gamma = pd.DataFrame(filas_resumen).sort_values("gamma").reset_index(drop=True)
    indice_max = df_gamma["f1_macro_mean"].idxmax()
    f1_max = float(df_gamma.loc[indice_max, "f1_macro_mean"])
    se_max = float(df_gamma.loc[indice_max, "f1_macro_se"])
    umbral_1se = f1_max - se_max
    df_gamma["within_1se_of_max"] = df_gamma["f1_macro_mean"] >= umbral_1se
    gamma_elegido = float(df_gamma.loc[df_gamma["within_1se_of_max"], "gamma"].min())
    df_gamma["selected"] = df_gamma["gamma"] == gamma_elegido

    report_dir = ensure_dir(report_dir)
    graph_dir = ensure_dir(graph_dir)
    df_gamma.to_csv(Path(report_dir) / "barrido_gamma.csv", index=False)

    apply_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    for metrica, etiqueta, color, marcador in [
        ("f1_macro", "F1-Macro", COLORS["seupurple"], "o"),
        ("balanced_accuracy", "Balanced Accuracy", COLORS["seuteal"], "s"),
        ("accuracy", "Accuracy", COLORS["lachaiseblue"], "^"),
        ("roc_auc_ovr", "ROC-AUC OvR", COLORS["seuamber"], "D"),
    ]:
        ax.errorbar(
            df_gamma["gamma"],
            df_gamma[f"{metrica}_mean"],
            yerr=df_gamma[f"{metrica}_se"],
            fmt=f"{marcador}-",
            color=color,
            label=f"{etiqueta} (±1 SE)",
            capsize=3,
        )
    ax.axvline(gamma_elegido, linestyle="--", color="black", alpha=0.7, label=f"γ* seleccionado = {gamma_elegido:.2f}")
    ax.set_xlabel("Exponente de ponderación γ")
    ax.set_ylabel("Puntuación media en CV (Train)")
    ax.set_title("Barrido de γ sobre Train (5-Fold CV) y regla de 1 Error Estándar")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save_figure(fig, Path(graph_dir) / "11_barrido_gamma.png")
    plt.close(fig)

    return gamma_elegido, df_gamma


def compare_balance_schemes(estimator, X: pd.DataFrame, y: pd.Series, k: int = 5, gamma: float = 0.5, random_state: int = SEED):
    """Compara esquemas de balanceo ajustando cada estrategia exclusivamente en el fold de entrenamiento."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    filas = []
    for esquema in ("baseline", "weighted", "oversampled"):
        metricas_por_fold = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            X_fold, y_fold = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            argumentos_fit = {}
            if esquema == "weighted" and gamma > 0.0:
                argumentos_fit["model__sample_weight"] = sample_weights(y_fold, gamma=gamma)
            elif esquema == "oversampled":
                X_fold, y_fold = oversample(X_fold, y_fold, method="random", random_state=random_state + fold)
            ajustado = clone(estimator)
            ajustado.fit(X_fold, y_fold, **argumentos_fit)
            prediccion = ajustado.predict(X_val)
            metricas_por_fold.append({
                "fold": fold,
                "f1_macro": float(f1_score(y_val, prediccion, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y_val, prediccion)),
                "f1_weighted": float(f1_score(y_val, prediccion, average="weighted", zero_division=0)),
                "accuracy": float(accuracy_score(y_val, prediccion)),
                "recall_frecuente": float(recall_score(y_val, prediccion, labels=["Frecuente"], average="macro", zero_division=0)),
                "recall_promocional": float(recall_score(y_val, prediccion, labels=["Promocional"], average="macro", zero_division=0)),
            })
        tabla = pd.DataFrame(metricas_por_fold)
        resumen = {"scheme": esquema, "gamma": float(gamma) if esquema == "weighted" else 0.0, "folds": metricas_por_fold}
        for metrica in tabla.columns[1:]:
            resumen[f"{metrica}_mean"] = float(tabla[metrica].mean())
            resumen[f"{metrica}_std"] = float(tabla[metrica].std(ddof=1))
            resumen[f"{metrica}_se"] = float(tabla[metrica].std(ddof=1) / np.sqrt(k))
        filas.append(resumen)
    return filas


def bootstrap_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = SEED,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Estima error estándar e IC 95 % en prueba mediante 1,000 remuestreos bootstrap."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    classes = np.asarray(classes)
    total = len(y_true)

    puntajes_puntuales = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr": float(roc_auc_score(y_true, y_proba, multi_class="ovr", labels=classes)),
    }

    generador = np.random.default_rng(random_state)
    semillas = generador.integers(0, 2**31 - 1, size=n_bootstraps)

    def _remuestreo(semilla: int):
        rng_local = np.random.default_rng(int(semilla))
        indices = rng_local.integers(0, total, size=total)
        yt, yp, ypr = y_true[indices], y_pred[indices], y_proba[indices]
        return (
            float(accuracy_score(yt, yp)),
            float(balanced_accuracy_score(yt, yp)),
            float(f1_score(yt, yp, average="macro", zero_division=0)),
            float(roc_auc_score(yt, ypr, multi_class="ovr", labels=classes)),
        )

    muestras = Parallel(n_jobs=n_jobs)(delayed(_remuestreo)(s) for s in semillas)
    matriz = np.asarray(muestras, dtype=float)
    metricas = ["accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr"]
    filas = []
    for i, nombre_metrica in enumerate(metricas):
        valores = matriz[:, i]
        filas.append({
            "metric": nombre_metrica,
            "point_estimate": puntajes_puntuales[nombre_metrica],
            "mean": float(valores.mean()),
            "std": float(valores.std(ddof=1)),
            "SE": float(valores.std(ddof=1)),
            "ci95_low": float(np.percentile(valores, 2.5)),
            "ci95_high": float(np.percentile(valores, 97.5)),
        })
    return pd.DataFrame(filas)


def resumir_estadistica_rejilla(results: pd.DataFrame, best_index: int, report_dir: Path) -> None:
    """Guarda el ranking estadístico completo de la rejilla y los intervalos del mejor modelo."""
    filas_ranking = []
    for indice_modelo, fila in results.iterrows():
        for metrica in ["f1_macro", "f1_weighted", "balanced_accuracy", "accuracy", "precision_macro", "recall_macro"]:
            columnas_folds = [c for c in results.columns if c.startswith("split") and c.endswith(f"_test_{metrica}")]
            valores = fila[columnas_folds].dropna().astype(float).to_numpy()
            if len(valores) == 0:
                continue
            media = float(valores.mean())
            desv = float(valores.std(ddof=1)) if len(valores) > 1 else 0.0
            error_estandar = float(desv / np.sqrt(len(valores)))
            margen = float(t.ppf(0.975, len(valores) - 1) * error_estandar) if len(valores) > 1 else 0.0
            filas_ranking.append({
                "model_index": indice_modelo,
                "metric": metrica,
                "mean": media,
                "std": desv,
                "SE": error_estandar,
                "CV": float(desv / media) if media else 0.0,
                "ci95_low": media - margen,
                "ci95_high": media + margen,
            })
    pd.DataFrame(filas_ranking).to_csv(report_dir / "ranking_estadistico_gridsearch.csv", index=False)

    filas_mejor = []
    for metrica in ["f1_macro", "balanced_accuracy", "accuracy", "roc_auc_ovr"]:
        valores = results.filter(regex=f"split\\d+_test_{metrica}").iloc[best_index].dropna().astype(float).values
        media = float(valores.mean())
        desv = float(valores.std(ddof=1)) if len(valores) > 1 else 0.0
        error_estandar = float(desv / np.sqrt(len(valores))) if valores.size else 0.0
        margen = float(t.ppf(0.975, len(valores) - 1) * error_estandar) if len(valores) > 1 else 0.0
        filas_mejor.append({
            "metric": metrica,
            "mean": media,
            "std": desv,
            "SE": error_estandar,
            "CV": float(desv / media) if media else 0.0,
            "ci95_low": media - margen,
            "ci95_high": media + margen,
        })
    pd.DataFrame(filas_mejor).to_csv(report_dir / "variacion_estadistica_mejor_modelo.csv", index=False)


def ejecutar_calibracion(best_estimator, validation: pd.DataFrame, test: pd.DataFrame, report_dir: Path) -> None:
    """Calibra el modelo congelado sobre validation, ajusta multiplicadores en validation y evalúa en test."""
    X_val, y_val = validation.drop(columns=[TARGET]), validation[TARGET]
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    metricas_base_test = probabilistic_metrics(best_estimator, X_test, y_test)
    metricas_base_val = probabilistic_metrics(best_estimator, X_val, y_val)
    mult_base, f1_val_base = tune_multipliers(best_estimator, X_val, y_val)
    pred_mult_base = apply_multipliers(best_estimator, X_test, mult_base)

    filas_calibracion = [{
        "method": "uncalibrated",
        **metricas_base_test,
        "validation_log_loss": metricas_base_val["log_loss"],
        "tuned_f1_macro_validation": float(f1_val_base),
        "tuned_f1_macro": float(f1_score(y_test, pred_mult_base, average="macro", zero_division=0)),
        "tuned_f1_macro_test": float(f1_score(y_test, pred_mult_base, average="macro", zero_division=0)),
        "tuned_accuracy_test": float(accuracy_score(y_test, pred_mult_base)),
        "tuned_balanced_accuracy_test": float(balanced_accuracy_score(y_test, pred_mult_base)),
        "worsened_log_loss": False,
        "multipliers": mult_base.tolist(),
    }]
    for metodo in ("sigmoid", "isotonic"):
        calibrado = calibrate(best_estimator, X_val, y_val, metodo)
        metricas_val = probabilistic_metrics(calibrado, X_val, y_val)
        metricas_test = probabilistic_metrics(calibrado, X_test, y_test)
        multiplicadores, f1_val_ajustado = tune_multipliers(calibrado, X_val, y_val)
        pred_test_mult = apply_multipliers(calibrado, X_test, multiplicadores)
        f1_test_ajustado = float(f1_score(y_test, pred_test_mult, average="macro", zero_division=0))
        metricas_test.update({
            "method": metodo,
            "validation_log_loss": metricas_val["log_loss"],
            "tuned_f1_macro_validation": float(f1_val_ajustado),
            "tuned_f1_macro": f1_test_ajustado,
            "tuned_f1_macro_test": f1_test_ajustado,
            "tuned_accuracy_test": float(accuracy_score(y_test, pred_test_mult)),
            "tuned_balanced_accuracy_test": float(balanced_accuracy_score(y_test, pred_test_mult)),
            "worsened_log_loss": bool(metricas_test["log_loss"] > metricas_base_test["log_loss"]),
            "multipliers": multiplicadores.tolist(),
        })
        joblib.dump(calibrado, MODELS / f"modelo_calibrado_{metodo}.joblib")
        filas_calibracion.append(metricas_test)
    pd.DataFrame(filas_calibracion).to_json(report_dir / "calibracion.json", orient="records", indent=2)


def run(
    train_path="train.csv",
    validation_path="validation.csv",
    test_path="test.csv",
    report_dir="resultados",
    graph_dir="graficas",
    fast=False,
    k=5,
    balance="weighted",
    gamma=None,
    calibrate_models=False,
    max_iter_values=(150, 300),
    n_jobs=-1,
):
    ensure_layout()
    train_path = first_existing(Path(train_path).name) if not Path(train_path).exists() else Path(train_path)
    validation_path = first_existing(Path(validation_path).name) if not Path(validation_path).exists() else Path(validation_path)
    test_path = first_existing(Path(test_path).name) if not Path(test_path).exists() else Path(test_path)
    train, validation, test = map(read_dataset, [train_path, validation_path, test_path])
    X, y = train.drop(columns=[TARGET]), train[TARGET]

    report_dir = ensure_dir(report_dir)
    graph_dir = ensure_dir(graph_dir)

    swept_gamma, _ = sweep_gamma(X, y, k=k, report_dir=report_dir, graph_dir=graph_dir, n_jobs=n_jobs)
    chosen_gamma = float(gamma) if gamma is not None else swept_gamma
    print(f"Gamma elegido por regla 1-SE en CV sobre train: {swept_gamma} (usado: {chosen_gamma})")

    pipe, grid = build_grid(fast=fast, max_iter_values=max_iter_values)
    n_combinaciones = count_grid_combinations(grid)
    print(f"Total combinaciones en la rejilla: {n_combinaciones} (fast={fast})")

    memory = Memory("./cache_pipeline", verbose=0)
    pipe.memory = memory
    search = GridSearchCV(
        pipe,
        grid,
        scoring=scoring(),
        refit="f1_macro",
        cv=get_stratified_cv(k),
        n_jobs=n_jobs,
        return_train_score=True,
        verbose=1,
        error_score=np.nan,
    )
    argumentos_fit = (
        {"model__sample_weight": sample_weights(y, chosen_gamma)}
        if (balance == "weighted" and chosen_gamma > 0.0)
        else {}
    )
    try:
        with parallel_backend("loky", n_jobs=n_jobs), threadpool_limits(limits=1):
            search.fit(X, y, **argumentos_fit)
    finally:
        memory.clear(warn=False)
        shutil.rmtree("./cache_pipeline", ignore_errors=True)

    search.best_estimator_.memory = None
    joblib.dump(search.best_estimator_, MODELS / "mejor_modelo_mlp.joblib")
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(report_dir / "resultados_gridsearch.csv", index=False)
    n_failed = int(results["mean_test_f1_macro"].isna().sum())
    print(f"Combinaciones fallidas (NaN): {n_failed} de {n_combinaciones}")

    resumir_estadistica_rejilla(results, search.best_index_, report_dir)

    # Una sola pasada de inferencia por partición, reutilizada en todos los reportes
    eval_train, y_train_true, y_train_pred, _ = evaluate(search.best_estimator_, train, "train")
    eval_val, y_val_true, y_val_pred, _ = evaluate(search.best_estimator_, validation, "validation")
    eval_test, y_test_true, y_test_pred, y_test_proba = evaluate(search.best_estimator_, test, "test")
    evaluaciones = [eval_train, eval_val, eval_test]

    for estrategia in ("most_frequent", "stratified"):
        dummy_pipe = Pipeline([
            ("features", DomainFeatureTransformer()),
            ("preprocessor", make_preprocessor("standard")),
            ("model", DummyClassifier(strategy=estrategia, random_state=SEED)),
        ])
        dummy_pipe.fit(X, y)
        dummy_eval, _, _, _ = evaluate(dummy_pipe, test, f"dummy_{estrategia}_test")
        evaluaciones.append(dummy_eval)

    eval_df = pd.DataFrame(evaluaciones)
    eval_df["generalization_gap_from_train"] = eval_df["f1_macro"].iloc[0] - eval_df["f1_macro"]
    eval_df.to_csv(report_dir / "evaluacion_splits.csv", index=False)

    reportes_por_particion = {
        "train": classification_report(y_train_true, y_train_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0),
        "validation": classification_report(y_val_true, y_val_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0),
        "test": classification_report(y_test_true, y_test_pred, labels=CLASS_ORDER, output_dict=True, zero_division=0),
    }
    Path(report_dir, "reportes_por_clase.json").write_text(
        json.dumps(reportes_por_particion, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    boot_df = bootstrap_test_metrics(
        y_test_true,
        y_test_pred,
        y_test_proba,
        search.best_estimator_.classes_,
        n_bootstraps=1000,
        random_state=SEED,
        n_jobs=n_jobs,
    )
    boot_df.to_csv(report_dir / "variacion_estadistica_test_bootstrap.csv", index=False)

    variables_originales = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
    val_x = validation[variables_originales].copy()
    val_y = validation[TARGET].copy()
    permutacion = permutation_importance(
        search.best_estimator_,
        val_x,
        val_y,
        scoring="f1_macro",
        n_repeats=10,
        random_state=SEED,
        n_jobs=min(4, n_jobs if n_jobs > 0 else 4),
    )
    importancia = pd.DataFrame({
        "feature": variables_originales,
        "importance_mean": permutacion.importances_mean,
        "importance_std": permutacion.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importancia.to_csv(report_dir / "importancia_permutacion.csv", index=False)

    balance_rows = compare_balance_schemes(search.best_estimator_, X, y, k=k, gamma=chosen_gamma)
    write_json(balance_rows, report_dir / "balance_comparison.json")

    if calibrate_models:
        ejecutar_calibracion(search.best_estimator_, validation, test, report_dir)

    write_json(
        {
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "k": int(k),
            "fast": bool(fast),
            "balance": balance,
            "gamma": float(chosen_gamma),
            "gamma_swept_1se": float(swept_gamma),
            "n_combinaciones": int(n_combinaciones),
            "n_failed_combinations": int(n_failed),
            "n_jobs": int(n_jobs),
        },
        report_dir / "mejor_modelo.json",
    )
    from src import generate_figures
    generate_figures.main()
    print(f"Mejor f1_macro CV: {search.best_score_:.4f}; parámetros: {search.best_params_}")
    return search.best_estimator_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--validation", default="validation.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--report-dir", default="resultados")
    parser.add_argument("--graph-dir", default="graficas")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--balance", choices=["baseline", "weighted"], default="weighted")
    parser.add_argument("--gamma", type=float, default=None, help="Fuerza el valor de gamma; si se omite usa la regla 1-SE")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    run(
        args.train,
        args.validation,
        args.test,
        args.report_dir,
        args.graph_dir,
        args.fast,
        args.k,
        args.balance,
        args.gamma,
        args.calibrate,
        n_jobs=args.n_jobs,
    )
