"""Entrenamiento leakage-safe de MLP y generación de evidencia experimental."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from joblib import Memory, parallel_backend
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, make_scorer, roc_curve, auc)
from sklearn.inspection import permutation_importance

from proyecto_utils import CLASS_ORDER, TARGET, ensure_dir, make_preprocessor, read_dataset, write_json
from crossVal import get_stratified_cv
from src.feature_engineering import DomainFeatureTransformer
from src.balancing import sample_weights, oversample
from src.calibration import calibrate, probabilistic_metrics, tune_multipliers
from src.plotting_theme import apply_theme, save_figure, CLASS_COLORS, COLORS
from src.paths import MODELS, RESULTS, FIGURES, ensure_layout, first_existing


def build_grid(fast=False):
    pipe = Pipeline([("features", DomainFeatureTransformer()),
                     ("preprocessor", make_preprocessor("standard")),
                     ("model", MLPClassifier(random_state=42, early_stopping=True,
                                             validation_fraction=0.1, n_iter_no_change=10, tol=1e-4))])
    if fast:
        grid = {"model__hidden_layer_sizes": [(64,)], "model__max_iter": [80],
                "model__learning_rate_init": [0.001], "model__learning_rate": ["adaptive"],
                "model__solver": ["adam"], "model__activation": ["relu"],
                "model__alpha": [0.001], "model__batch_size": [256]}
    else:
        grid = [
            {"model__hidden_layer_sizes": [(64,), (128, 64), (128, 64, 32)],
             "model__max_iter": [150, 300], "model__learning_rate_init": [0.001, 0.005, 0.01],
             "model__learning_rate": ["constant", "adaptive"], "model__solver": ["adam"],
             "model__activation": ["relu", "tanh"], "model__alpha": [1e-4, 1e-3, 1e-2],
             "model__batch_size": [256, 512]},
            {"model__hidden_layer_sizes": [(64,), (128, 64)], "model__max_iter": [150, 300],
             "model__learning_rate_init": [0.001, 0.005], "model__learning_rate": ["constant", "adaptive"],
             "model__solver": ["sgd"], "model__momentum": [0.9, 0.95],
             "model__nesterovs_momentum": [True], "model__activation": ["relu", "tanh"],
             "model__alpha": [1e-4, 1e-3], "model__batch_size": [256, 512]},
        ]
    return pipe, grid


def scoring():
    return {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_weighted": make_scorer(f1_score, average="weighted"),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "roc_auc_ovr": "roc_auc_ovr",
    }


def evaluate(model, df, split):
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    pred, proba = model.predict(X), model.predict_proba(X)
    result = {"split": split, "accuracy": accuracy_score(y, pred),
              "f1_macro": f1_score(y, pred, average="macro"), "f1_weighted": f1_score(y, pred, average="weighted"),
              "balanced_accuracy": balanced_accuracy_score(y, pred),
              "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
              "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
              "roc_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", labels=model.classes_)}
    return result, y, pred, proba


def compare_balance_schemes(estimator, X, y, k=5, random_state=42):
    """Compara balanceos entrenando cada esquema exclusivamente en cada fold."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    rows = []
    for scheme in ("baseline", "weighted", "oversampled"):
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            X_fold, y_fold = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            fit_kwargs = {}
            if scheme == "weighted":
                fit_kwargs["model__sample_weight"] = sample_weights(y_fold, gamma=0.75)
            elif scheme == "oversampled":
                X_fold, y_fold = oversample(X_fold, y_fold, method="random", random_state=random_state + fold)
            fitted = clone(estimator)
            fitted.fit(X_fold, y_fold, **fit_kwargs)
            prediction = fitted.predict(X_val)
            fold_metrics.append({
                "fold": fold,
                "f1_macro": f1_score(y_val, prediction, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y_val, prediction),
                "f1_weighted": f1_score(y_val, prediction, average="weighted"),
                "recall_frecuente": recall_score(y_val, prediction, labels=["Frecuente"], average="macro", zero_division=0),
                "recall_promocional": recall_score(y_val, prediction, labels=["Promocional"], average="macro", zero_division=0),
            })
        frame = pd.DataFrame(fold_metrics)
        summary = {"scheme": scheme, "folds": fold_metrics}
        for metric in frame.columns[1:]:
            summary[f"{metric}_mean"] = float(frame[metric].mean())
            summary[f"{metric}_std"] = float(frame[metric].std(ddof=1))
            summary[f"{metric}_se"] = float(frame[metric].std(ddof=1) / np.sqrt(k))
        rows.append(summary)
    return rows


def make_plots(model, test_df, cv_results, out_dir):
    apply_theme()
    out_dir = ensure_dir(out_dir)
    estimator = model.named_steps["model"]
    if hasattr(estimator, "loss_curve_"):
        plt.figure(figsize=(8, 5)); plt.plot(estimator.loss_curve_); plt.xlabel("Época"); plt.ylabel("Pérdida")
        plt.title("Curva de pérdida del mejor MLP");         fig = plt.gcf(); plt.tight_layout(); save_figure(fig, out_dir / "04_curvas_aprendizaje_convergencia.png"); plt.close()
    if hasattr(estimator, "validation_scores_") and estimator.validation_scores_:
        plt.figure(figsize=(8, 5)); plt.plot(estimator.validation_scores_); plt.xlabel("Época"); plt.ylabel("Score validación")
        fig = plt.gcf(); plt.tight_layout(); save_figure(fig, out_dir / "04_curvas_aprendizaje_convergencia_validation.png"); plt.close()
    frame = pd.DataFrame(cv_results)
    if "mean_test_f1_macro" in frame:
        top = frame.nlargest(min(20, len(frame)), "mean_test_f1_macro")
        plt.figure(figsize=(10, 6)); sns.barplot(data=top, x="mean_test_f1_macro", y="rank_test_f1_macro")
        fig = plt.gcf(); plt.tight_layout(); save_figure(fig, out_dir / "03_evolucion_rejilla_paralela.png"); plt.close()
    fold_values = []
    for metric in ("f1_macro", "balanced_accuracy"):
        cols = [c for c in frame.columns if c.startswith("split") and c.endswith(f"_test_{metric}")]
        if cols:
            for value in frame.loc[frame["rank_test_f1_macro"].idxmin(), cols].astype(float):
                fold_values.append({"metric": metric, "score": value})
    if fold_values:
        plt.figure(figsize=(8, 5)); sns.boxplot(data=pd.DataFrame(fold_values), x="metric", y="score")
        plt.title("Variación estadística del mejor modelo entre folds")
        fig = plt.gcf(); plt.tight_layout(); save_figure(fig, out_dir / "07_variacion_estadistica_boxplots.png"); plt.close()
    _, y, pred, proba = evaluate(model, test_df, "test")
    cm = confusion_matrix(y, pred, labels=model.classes_, normalize="true")
    plt.figure(figsize=(7, 6)); sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel("Predicción"); plt.ylabel("Real");     fig = plt.gcf(); plt.tight_layout(); save_figure(fig, out_dir / "08_matriz_confusion_test.png"); plt.close()
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(model.classes_):
        binary = (y == cls).astype(int); fpr, tpr, _ = roc_curve(binary, proba[:, i])
        plt.plot(fpr, tpr, label=f"{cls} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray"); plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR")
    fig = plt.gcf(); plt.tight_layout(); save_figure(fig, out_dir / "09_curvas_roc_pr_multiclase.png"); plt.close()


def run(train_path="train.csv", validation_path="validation.csv", test_path="test.csv",
        report_dir="resultados", graph_dir="graficas", fast=False, k=5,
        balance="baseline", calibrate_models=False, n_jobs=-1):
    ensure_layout()
    train_path = first_existing(Path(train_path).name) if not Path(train_path).exists() else Path(train_path)
    validation_path = first_existing(Path(validation_path).name) if not Path(validation_path).exists() else Path(validation_path)
    test_path = first_existing(Path(test_path).name) if not Path(test_path).exists() else Path(test_path)
    train, validation, test = map(read_dataset, [train_path, validation_path, test_path])
    X, y = train.drop(columns=[TARGET]), train[TARGET]
    pipe, grid = build_grid(fast)
    memory = Memory("./cache_pipeline", verbose=0)
    pipe.memory = memory
    search = GridSearchCV(pipe, grid, scoring=scoring(), refit="f1_macro", cv=get_stratified_cv(k),
                          n_jobs=n_jobs, return_train_score=True, verbose=1, error_score="raise")
    fit_kwargs = {"model__sample_weight": sample_weights(y, 0.75)} if balance == "weighted" else {}
    with parallel_backend("loky", n_jobs=n_jobs), threadpool_limits(limits=1):
        search.fit(X, y, **fit_kwargs)
    report_dir = ensure_dir(report_dir)
    joblib.dump(search.best_estimator_, MODELS / "mejor_modelo_mlp.joblib")
    results = pd.DataFrame(search.cv_results_)
    results.to_csv(report_dir / "resultados_gridsearch.csv", index=False)
    ranking_rows = []
    for row_index, row in results.iterrows():
        for metric in ["f1_macro", "f1_weighted", "balanced_accuracy", "precision_macro", "recall_macro"]:
            values = row[[c for c in results.columns if c.startswith("split") and c.endswith(f"_test_{metric}")]].dropna().astype(float).to_numpy()
            if len(values) == 0:
                continue
            mean, std = values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0
            se = std / np.sqrt(len(values))
            margin = t.ppf(0.975, len(values) - 1) * se if len(values) > 1 else 0.0
            ranking_rows.append({"model_index": row_index, "metric": metric, "mean": mean, "std": std,
                                 "SE": se, "CV": std / mean if mean else 0.0,
                                 "ci95_low": mean - margin, "ci95_high": mean + margin})
    pd.DataFrame(ranking_rows).to_csv(report_dir / "ranking_estadistico_gridsearch.csv", index=False)
    evaluations = [evaluate(search.best_estimator_, frame, name)[0] for name, frame in
                    [("train", train), ("validation", validation), ("test", test)]]
    eval_df = pd.DataFrame(evaluations)
    eval_df["generalization_gap_from_train"] = eval_df["f1_macro"].iloc[0] - eval_df["f1_macro"]
    eval_df.to_csv(report_dir / "evaluacion_splits.csv", index=False)
    val_x, val_y = validation.drop(columns=[TARGET]), validation[TARGET]
    feature_step = search.best_estimator_.named_steps["features"]
    prep_step = search.best_estimator_.named_steps["preprocessor"]
    transformed_validation = prep_step.transform(feature_step.transform(val_x))
    names = prep_step.get_feature_names_out()
    permutation = permutation_importance(
        search.best_estimator_.named_steps["model"], transformed_validation, val_y,
        scoring="f1_macro", n_repeats=2, random_state=42, n_jobs=min(2, n_jobs if n_jobs > 0 else 2),
    )
    importance = pd.DataFrame({"feature": names, "importance_mean": permutation.importances_mean,
                               "importance_std": permutation.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(report_dir / "importancia_permutacion.csv", index=False)
    reports = {}
    for name, frame in [("train", train), ("validation", validation), ("test", test)]:
        result, y_true, pred, _ = evaluate(search.best_estimator_, frame, name)
        reports[name] = classification_report(y_true, pred, labels=CLASS_ORDER, output_dict=True, zero_division=0)
    Path(report_dir, "reportes_por_clase.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    fold_rows = []
    for metric in ["f1_macro", "balanced_accuracy"]:
        values = results.filter(regex=f"split\\d+_test_{metric}").iloc[search.best_index_].dropna().astype(float).values
        mu, sigma = values.mean(), values.std(ddof=1) if len(values) > 1 else 0
        se = sigma / np.sqrt(len(values)) if values.size else 0
        margin = t.ppf(0.975, len(values)-1) * se if len(values) > 1 else 0
        fold_rows.append({"metric": metric, "mean": mu, "std": sigma, "SE": se, "CV": sigma / mu if mu else 0,
                          "ci95_low": mu-margin, "ci95_high": mu+margin})
    pd.DataFrame(fold_rows).to_csv(report_dir / "variacion_estadistica_mejor_modelo.csv", index=False)
    balance_rows = compare_balance_schemes(search.best_estimator_, X, y, k=k)
    write_json(balance_rows, report_dir / "balance_comparison.json")
    make_plots(search.best_estimator_, test, search.cv_results_, graph_dir)
    if calibrate_models:
        calibration_rows = []
        for method in ("sigmoid", "isotonic"):
            calibrated = calibrate(search.best_estimator_, validation.drop(columns=[TARGET]), validation[TARGET], method)
            metrics = probabilistic_metrics(calibrated, test.drop(columns=[TARGET]), test[TARGET])
            multipliers, tuned_f1 = tune_multipliers(calibrated, validation.drop(columns=[TARGET]), validation[TARGET])
            metrics.update({"method": method, "tuned_f1_macro": tuned_f1, "multipliers": multipliers.tolist()})
            joblib.dump(calibrated, MODELS / f"modelo_calibrado_{method}.joblib")
            calibration_rows.append(metrics)
        pd.DataFrame(calibration_rows).to_json(report_dir / "calibracion.json", orient="records", indent=2)
    write_json({"best_params": search.best_params_, "best_score": search.best_score_, "k": k,
                "fast": fast, "balance": balance, "n_jobs": n_jobs},
               report_dir / "mejor_modelo.json")
    print(f"Mejor f1_macro CV: {search.best_score_:.4f}; parámetros: {search.best_params_}")
    return search.best_estimator_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv"); parser.add_argument("--validation", default="validation.csv")
    parser.add_argument("--test", default="test.csv"); parser.add_argument("--report-dir", default="reportes")
    parser.add_argument("--graph-dir", default="graficas"); parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--balance", choices=["baseline", "weighted"], default="baseline")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    run(args.train, args.validation, args.test, args.report_dir, args.graph_dir, args.fast, args.k,
        args.balance, args.calibrate, args.n_jobs)
