from dataclasses import dataclass, field


@dataclass()
class PredictParams:
    path_to_data: str
    path_to_model: str
    output_path: str