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
    output_file: str
    voc_size: int