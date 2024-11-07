from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngstionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataPreprocessConfig:
    root_dir: Path
    source_dir: Path
    voc_size: int
    text_preprocessor: str
    target_preprocessor: str
    max_length:int

@dataclass
class ModelConfig:
    root_dir: Path
    model_path: Path
    batch: int
    epochs: int
    Voc_size: int
    Max_features: int
    optimizer: str
    loss: str
    metrics: list
    max_length: int
