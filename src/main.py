import hydra
from omegaconf import DictConfig
from src.data.openbb_dataset import OpenBBDataModule
from src.namespaces.data_config import DataConfig
from openbb import obb

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Apply OpenBB settings
    obb.user.preferences.output_type = cfg.openbb.output_type

    data_config = DataConfig(**cfg.data)
    data_module = OpenBBDataModule(data_config)
    data_module.prepare_data()
    data_module.setup()

    # Your training code here
    ...

if __name__ == "__main__":
    main()
