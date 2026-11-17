"""Scikit-learn closed-set identity classification baseline."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def build_random_forest(n_estimators: int = 300, random_state: int = 42) -> Pipeline:
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight="balanced_subsample",
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
