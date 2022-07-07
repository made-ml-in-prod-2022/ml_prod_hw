from dataclasses import dataclass, field


@dataclass()
class TrainParams:
    SEED: int = field(default=322)
    input_size: int = field(default=13)
    num_classes: int = field(default=2)
    learning_rate: float = field(default=0.001)
    batch_size: int = field(default=1)
    num_epochs: int = field(default=200)
    weight_decay: int = field(default=0.0001)
