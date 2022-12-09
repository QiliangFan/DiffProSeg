import pretty_errors

import yaml
import os
import torch
from pytorch_lightning import Trainer, LightningModule, LightningDataModule
from pytorch_lightning.loggers import CSVLogger

from models import DiffusionModel, CondImageUNet
from utils import SeedContext
from data import Promise12Dataset

def train_loop(trainer: Trainer, module: LightningModule, data: LightningDataModule):
    trainer.fit(module, datamodule=data)

def test_loop(trainer: Trainer, module: LightningModule, data: LightningDataModule):
    trainer.test(module, datamodule=data)

def main():
    diffusion_config = global_config["diffusion"]
    train_config = global_config["train"]

    
    for fold_idx in range(5):
        with SeedContext(108):
            promise12_data = Promise12Dataset(data_root, fold_idx)
            epsilon_theta = CondImageUNet(diffusion_config["num_steps"])
            diffusion_model = DiffusionModel(epsilon_theta, diffusion_config["num_steps"], train_config)

            # Trainer
            logger = CSVLogger("logs", "default")
            trainer = Trainer(
                logger=logger,
                gpus=1 if torch.cuda.device_count() > 0 else 0,
                max_epochs=train_config["epochs"],
                benchmark=True,
            )

            train_loop(trainer, diffusion_model, promise12_data)
            test_loop(trainer, diffusion_model, promise12_data)

if __name__ == "__main__":
    data_root = "/home/fanqiliang/data/processed_data/hist_32"
    global_config = yaml.full_load(open(os.path.join("config", "config.yaml")))

    main()
