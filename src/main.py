import hydra
from omegaconf import DictConfig
from src.data.openbb_dataset import OpenBBDataModule

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    data_module = OpenBBDataModule(cfg.data)
    data_module.prepare_data()
    data_module.setup()

    # Your training code here
    ...

if __name__ == "__main__":
    main()
