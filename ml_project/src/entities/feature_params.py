from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    numerical_grouped_features: Optional[List[str]]
    target_col: Optional[str]