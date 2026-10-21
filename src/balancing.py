"""Esquemas de balanceo; el remuestreo se invoca sobre cada fold de train."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

def class_weights(y, gamma=1.0):
    classes = np.unique(y)
    balanced = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, balanced ** gamma))

def sample_weights(y, gamma=1.0):
    weights = class_weights(y, gamma)
    return np.asarray([weights[value] for value in y], dtype=float)

def oversample(X, y, method="random", random_state=42):
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        sampler = SMOTE(random_state=random_state) if method == "smote" else RandomOverSampler(random_state=random_state)
        return sampler.fit_resample(X, y)
    except ImportError:
        frame = X.copy()
        frame["_target"] = np.asarray(y)
        parts = [group.sample(max(frame["_target"].value_counts()), replace=True, random_state=random_state)
                 for _, group in frame.groupby("_target")]
        result = pd.concat(parts).sample(frac=1, random_state=random_state)
        return result.drop(columns="_target"), result["_target"]
