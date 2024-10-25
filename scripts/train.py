import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from alphadl.data.openbb_dataset import OpenBBDataModule
from alphadl.models import LSTM, XGBoost, CatBoost, MLP

@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    # Initialize data module
    data_module = OpenBBDataModule(cfg.data)

    # Initialize model
    if cfg.model.name == "lstm":
        model = LSTM(cfg.model)
    elif cfg.model.name == "xgboost":
        model = XGBoost(cfg.model)
    elif cfg.model.name == "catboost":
        model = CatBoost(cfg.model)
    elif cfg.model.name == "mlp":
        model = MLP(cfg.model)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")

    # Initialize WandB logger
    wandb_logger = WandbLogger(project=cfg.wandb.project, entity=cfg.wandb.entity)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        gpus=cfg.trainer.gpus,
        logger=wandb_logger,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Save the model
    trainer.save_checkpoint(f"{cfg.model.name}_model.ckpt")

if __name__ == "__main__":
    train()
