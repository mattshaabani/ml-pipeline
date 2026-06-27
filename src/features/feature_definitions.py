"""
src/features/feature_definitions.py

Defines the feature schema for the Adult Income dataset.
This is the single source of truth for what features exist,
their types, and how they should be processed.

In a production Feast setup, these would be registered as
FeatureView objects in a feature registry. Here we define
them as Python dataclasses that drive our preprocessing.

Usage:
    from src.features.feature_definitions import FEATURE_SCHEMA
    print(FEATURE_SCHEMA.numerical_features)
"""

from dataclasses import dataclass, field
from src.utils.config import settings


@dataclass
class FeatureSchema:
    """
    Defines the complete feature schema for the pipeline.

    Separating schema definition from transformation logic means:
    - Adding a new feature = add it here, transformers pick it up
    - Schema changes are auditable in git
    - Easy to validate incoming data against expected schema
    """
    numerical_features:   list[str]
    categorical_features: list[str]
    target:               str
    features_to_drop:     list[str] = field(default_factory=list)

    @property
    def all_features(self) -> list[str]:
        return self.numerical_features + self.categorical_features

    @property
    def feature_count(self) -> int:
        return len(self.all_features)


# The canonical feature schema for this project
# All pipeline components reference this object
FEATURE_SCHEMA = FeatureSchema(
    numerical_features=settings.features.numerical,
    categorical_features=settings.features.categorical,
    target=settings.features.target,
    # fnlwgt is a census weight — not a real predictive feature
    # education_num and education carry the same information
    # we keep education_num (numerical) and drop education (categorical)
    features_to_drop=["fnlwgt", "education"],
)


@dataclass
class FeatureStats:
    """
    Statistics computed from training data.
    Saved alongside the model so inference uses the same
    statistics (not test set statistics — that would be leakage).
    """
    numerical_means:   dict[str, float] = field(default_factory=dict)
    numerical_stds:    dict[str, float] = field(default_factory=dict)
    numerical_medians: dict[str, float] = field(default_factory=dict)
    categorical_modes: dict[str, str]   = field(default_factory=dict)
    categorical_vocab: dict[str, list]  = field(default_factory=dict)
    n_samples:         int              = 0