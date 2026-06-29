"""
src/training/models.py

Model definitions for the Adult Income classification task.

Four models covering the spectrum from interpretable to powerful:
    1. Logistic Regression  — baseline, fully interpretable
    2. Random Forest        — ensemble, good out-of-the-box
    3. XGBoost              — gradient boosting, often best on tabular
    4. LightGBM             — faster gradient boosting, similar accuracy

Usage:
    from src.training.models import get_model, MODEL_REGISTRY
    model = get_model("xgboost", n_estimators=100, max_depth=6)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_logistic_regression(**kwargs) -> LogisticRegression:
    """
    Logistic Regression — the interpretable baseline.

    Despite its simplicity, logistic regression is hard to beat
    when features are well-engineered and the decision boundary
    is roughly linear in feature space.

    The model outputs:
        p(y=1|x) = sigmoid(w·x + b) = 1 / (1 + exp(-(w·x + b)))

    Training minimizes cross-entropy loss:
        L = -Σ [y·log(p) + (1-y)·log(1-p)]

    Regularization parameter C = 1/λ:
        High C → less regularization → can overfit
        Low C  → strong regularization → simpler model
    """
    defaults = {
        "C":          1.0,
        "max_iter":   1000,
        "solver":     "lbfgs",
        "random_state": 42,
    }
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def get_random_forest(**kwargs) -> RandomForestClassifier:
    """
    Random Forest — bagging ensemble of decision trees.

    Each tree is trained on a bootstrap sample (sampling with replacement)
    and at each split considers only a random subset of features.
    Final prediction: majority vote across all trees.

    Key hyperparameters:
        n_estimators: more trees = more stable, diminishing returns after ~200
        max_depth:    deeper = more complex, higher variance
        max_features: "sqrt" is standard — sqrt(n_features) per split
    """
    defaults = {
        "n_estimators": 100,
        "max_depth":    10,
        "random_state": 42,
        "n_jobs":       -1,
    }
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def get_xgboost(**kwargs):
    """
    XGBoost — gradient boosted trees with second-order optimization.

    Key hyperparameters:
        n_estimators:    number of trees
        max_depth:       tree depth (3-10 typical)
        learning_rate:   shrinkage per tree (0.01-0.3)
        subsample:       fraction of samples per tree (prevents overfitting)
        colsample_bytree: fraction of features per tree
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        raise ImportError("Run: pip install xgboost")

    defaults = {
        "n_estimators":     100,
        "max_depth":        6,
        "learning_rate":    0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "random_state":     42,
        "eval_metric":      "logloss",
        "verbosity":        0,
        "n_jobs":           -1,
    }
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def get_lightgbm(**kwargs):
    """
    LightGBM — leaf-wise gradient boosting, faster than XGBoost.

    Key difference from XGBoost:
        XGBoost grows trees level-wise (all nodes at depth k before depth k+1)
        LightGBM grows leaf-wise (always splits the leaf with highest gain)

        Leaf-wise = faster convergence but needs max_depth to prevent overfit.

    num_leaves controls tree complexity:
        A balanced tree of depth d has 2^d leaves.
        LightGBM trees are unbalanced so num_leaves < 2^max_depth.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("Run: pip install lightgbm")

    defaults = {
        "n_estimators":  100,
        "max_depth":     6,
        "learning_rate": 0.1,
        "num_leaves":    31,
        "subsample":     0.8,
        "random_state":  42,
        "verbosity":     -1,
    }
    defaults.update(kwargs)
    return lgb.LGBMClassifier(**defaults)


# Registry mapping config names to factory functions
MODEL_REGISTRY = {
    "logistic_regression": get_logistic_regression,
    "random_forest":       get_random_forest,
    "xgboost":             get_xgboost,
    "lightgbm":            get_lightgbm,
}


def get_model(model_name: str, **kwargs):
    """
    Factory function — get a model by name.

    Usage:
        model = get_model("xgboost", n_estimators=200, max_depth=8)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)