from dataclasses import dataclass

@dataclass
class DataConfig:
    ticker: str
    start_date: str
    end_date: str
    sequence_length: int
    batch_size: int
    market: str
    interval: str
    cache_dir: str
    use_cache: bool
    provider: str
