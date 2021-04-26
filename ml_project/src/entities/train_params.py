from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="RandomForestClassifier")
    n_estimators: int = field(default=100)
    random_state: int = field(default=42)
    max_iter: int = field(default=20)
