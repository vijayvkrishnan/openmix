"""
ML model scorer — trained stability prediction.

Wraps a trained classifier/regressor to produce StabilityScore objects
that the experiment framework can use.

Usage:
    # Train and save a model
    scorer = ModelScorer.train(dataset, feature_fn=tier1_features)
    scorer.save("models/shampoo_stability.pkl")

    # Load and use
    scorer = ModelScorer.load("models/shampoo_stability.pkl")
    exp = Experiment.from_file("exp.yaml", evaluate=scorer)
"""

from __future__ import annotations

import pickle
from pathlib import Path

try:
    import joblib
    _USE_JOBLIB = True
except ImportError:
    _USE_JOBLIB = False
from typing import Callable

import numpy as np

from openmix.schema import Formula
from openmix.score import StabilityScore
from openmix.scorers.base import Scorer


class ModelScorer(Scorer):
    """
    Trained model as evaluation function.

    The model predicts stability probability (0-1), which is mapped
    to a StabilityScore (0-100) with decomposed sub-scores where possible.
    """

    def __init__(
        self,
        model,
        feature_fn: Callable,
        feature_names: list[str] | None = None,
        domain: str = "general",
        threshold: float = 0.5,
    ):
        self.model = model
        self.feature_fn = feature_fn
        self.feature_names = feature_names
        self.domain = domain
        self.threshold = threshold

    def __call__(self, formula: Formula) -> StabilityScore:
        try:
            features = self.feature_fn(formula)
            if features is None:
                return self._fallback_score()

            X = np.array([features], dtype=np.float32)

            # Get probability if available, else use raw prediction
            if hasattr(self.model, 'predict_proba'):
                prob = float(self.model.predict_proba(X)[0, 1])
            else:
                pred = float(self.model.predict(X)[0])
                prob = max(0, min(1, pred))

            return self._prob_to_score(prob)

        except Exception:
            return self._fallback_score()

    def _prob_to_score(self, prob: float) -> StabilityScore:
        """Convert a stability probability to a decomposed score."""
        total = round(prob * 100, 1)

        # Distribute across sub-scores proportionally
        # This is approximate — a better model would predict each sub-score
        result = StabilityScore(
            total=total,
            compatibility=round(prob * 35, 1),
            ph_suitability=round(prob * 25, 1),
            emulsion_balance=round(prob * 20, 1),
            formula_integrity=round(min(prob * 1.2, 1.0) * 10, 1),
            system_completeness=round(min(prob * 1.1, 1.0) * 10, 1),
        )

        if prob >= 0.8:
            result.bonuses.append(f"Model predicts {prob:.0%} stability probability")
        elif prob >= 0.5:
            result.bonuses.append(f"Model predicts {prob:.0%} stability (moderate confidence)")
        else:
            result.penalties.append(f"Model predicts {prob:.0%} stability (likely unstable)")

        return result

    def _fallback_score(self) -> StabilityScore:
        """Return when model can't score (unknown ingredients, etc.)."""
        result = StabilityScore(total=50.0)
        result.penalties.append("Model could not score this formula (unknown domain)")
        return result

    @property
    def name(self) -> str:
        model_type = type(self.model).__name__
        return f"model/{model_type}/{self.domain}"

    @property
    def description(self) -> str:
        return f"Trained {type(self.model).__name__} on {self.domain} domain"

    def save(self, path: str | Path):
        """Save the scorer (model + feature function + metadata)."""
        data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "domain": self.domain,
            "threshold": self.threshold,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if _USE_JOBLIB:
            joblib.dump(data, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(data, f)

    @classmethod
    def load(cls, path: str | Path, feature_fn: Callable | None = None) -> ModelScorer:
        """Load a saved scorer."""
        if _USE_JOBLIB:
            data = joblib.load(path)
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)

        if feature_fn is None:
            raise ValueError(
                "feature_fn is required when loading a ModelScorer. "
                "Pass the same feature function used during training."
            )

        return cls(
            model=data["model"],
            feature_fn=feature_fn,
            feature_names=data.get("feature_names"),
            domain=data.get("domain", "general"),
            threshold=data.get("threshold", 0.5),
        )

    @classmethod
    def train(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_fn: Callable,
        feature_names: list[str] | None = None,
        domain: str = "general",
        model_type: str = "xgboost",
    ) -> ModelScorer:
        """
        Train a model scorer from data.

        Args:
            X_train: Feature matrix (n_samples, n_features)
            y_train: Binary labels (1=stable, 0=unstable)
            feature_fn: Function to extract features from a Formula
            feature_names: Names for interpretability
            domain: Domain label
            model_type: "xgboost" or "random_forest"
        """
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                random_state=42, eval_metric="logloss",
            )
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=42,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        model.fit(X_train, y_train)

        return cls(
            model=model,
            feature_fn=feature_fn,
            feature_names=feature_names,
            domain=domain,
        )
