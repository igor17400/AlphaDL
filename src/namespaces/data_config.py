from dataclasses import dataclass

@dataclass
class DataConfig:
    ticker: str
    train_date: str
    val_date: str
    test_date: str
    sequence_length: int
    batch_size: int
    market: str
    interval: str
    cache_dir: str
    use_cache: bool
    provider: str
    openbb_output_type: str
