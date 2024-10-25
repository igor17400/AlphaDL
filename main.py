import hydra
from omegaconf import DictConfig
import pyrootutils
from typing import Optional

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from newsreclib import utils
from train_ import train
from eval_ import evaluate

log = utils.get_pylogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="main.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    utils.extras(cfg)

    # choose between train and eval mode
    if cfg.mode == "train":
        log.info("Starting training mode")
        metric_dict, _ = train(cfg)
        
        # safely retrieve metric value for hydra-based hyperparameter optimization
        metric_value = utils.get_metric_value(
            metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
        )
        return metric_value

    elif cfg.mode == "eval":
        log.info("Starting evaluation mode")
        evaluate(cfg)

    else:
        log.error(f"Invalid mode: {cfg.mode}. Choose either 'train' or 'eval'.")
        return None

if __name__ == "__main__":
    main()
