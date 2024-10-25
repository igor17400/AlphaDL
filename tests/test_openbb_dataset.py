import sys
import os

import pytest
from omegaconf import OmegaConf
from hydra import initialize, compose

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.openbb_dataset import OpenBBDataModule

@pytest.fixture(scope="module")
def hydra_config():
    with initialize(config_path="../configs"):
        cfg = compose(config_name="config", overrides=["data=data", "data.market=us_daily"])
    return cfg.data

def test_data_download(hydra_config):
    data_module = OpenBBDataModule(hydra_config)
    data_module.prepare_data()
    
    assert data_module.data is not None
    assert len(data_module.data) > 0
    assert 'Close' in data_module.data.columns
    
    # Check if cache file was created
    cache_file = os.path.join(hydra_config.cache_dir, f"{hydra_config.ticker}_{hydra_config.market}_{hydra_config.interval}.csv")
    assert os.path.exists(cache_file)

def test_data_loading_from_cache(hydra_config):
    # First, prepare data to create cache
    data_module = OpenBBDataModule(hydra_config)
    data_module.prepare_data()
    
    # Now, set use_cache to True and load again
    hydra_config_with_cache = OmegaConf.create(hydra_config)
    hydra_config_with_cache.use_cache = True
    cached_data_module = OpenBBDataModule(hydra_config_with_cache)
    cached_data_module.prepare_data()
    
    assert cached_data_module.data is not None
    assert len(cached_data_module.data) > 0
    assert 'Close' in cached_data_module.data.columns

@pytest.mark.parametrize("market_config", [
    "us_daily",
    "us_minute",
    "brazil_daily",
    "japan_daily",
])
def test_different_markets_and_intervals(hydra_config, market_config):
    with initialize(config_path="../configs"):
        cfg = compose(config_name="config", overrides=[f"data.market={market_config}"])
    
    data_module = OpenBBDataModule(cfg.data)
    data_module.prepare_data()
    
    assert data_module.data is not None
    assert len(data_module.data) > 0
    assert 'Close' in data_module.data.columns

def test_setup_method(hydra_config):
    data_module = OpenBBDataModule(hydra_config)
    data_module.prepare_data()
    data_module.setup()
    
    assert data_module.train_data is not None
    assert data_module.val_data is not None
    assert len(data_module.train_data) > len(data_module.val_data)
