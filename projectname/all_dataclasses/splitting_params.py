from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    manual_seed : int = field(default=322)
    train_percentage: int = field(default=75)
