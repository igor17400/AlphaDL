import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import importlib

from alphadl.data.openbb_dataset import OpenBBDataModule

@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    # Initialize data module
    data_module = OpenBBDataModule(cfg.data)

    # Dynamically import the model class
    model_module = importlib.import_module(f"alphadl.models.{cfg.model.name}")
    model_class = getattr(model_module, cfg.model.class_name)
    model = model_class(**cfg.model.params)

    # Dynamically import the loss function
    loss_module = importlib.import_module(f"losses.{cfg.loss.name}")
    loss_function = getattr(loss_module, cfg.loss.function_name)

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

    # Example of using the loss function
    # Assuming you have prediction, ground_truth, base_price, mask, and batch_size available
    # loss, reg_loss, rank_loss, return_ratio = loss_function(prediction, ground_truth, base_price, mask, batch_size, **cfg.loss.params)

    # Save the model
    trainer.save_checkpoint(f"{cfg.model.name}_model.ckpt")

if __name__ == "__main__":
    train()
