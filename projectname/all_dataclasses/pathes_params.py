from dataclasses import dataclass, field


@dataclass()
class PathesParams:
    input_data_path: str 
    output_model_path: str
    metric_path: str
    data_url: str