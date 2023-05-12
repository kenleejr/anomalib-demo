# FORM VARIABLES
PROJECT_NAME = "anomalib-demo" 
ENTITY = "cvproject-trial-team"

import logging
import warnings
import tempfile
import json
from pathlib import Path

import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
import torch

from omegaconf import OmegaConf
from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger

def train():
    """Train an anomaly classification or segmentation model based on a provided configuration file."""
    # Configure the wandb run and get wandb config
    logger = logging.getLogger("anomalib")
    run = wandb.init(project=PROJECT_NAME, entity=ENTITY,
                           job_type='training',
                           config={
                            "log_level": "INFO",
                            "show_images": True,
                            "run_name": "MVTec-transistor-train",
                            "dataset": {
                                "name": "MVTec-transistor",
                                "format": "folder",
                                "dataset-artifact": "MVTec-transistor:latest",
                                "root": "./artifacts/",
                                "normal_dir": "normal_dir/",
                                "abnormal_dir": "abnormal_dir/",
                                "normal_test_dir": "normal_test_dir/",
                                "mask_dir": "mask_dir/",
                                "extensions": ".png",
                                "task": "segmentation",
                                "train_batch_size": 32,
                                "test_batch_size": 32,
                                "num_workers": 8,
                                "image_size": 256,
                                "center_crop": 224,
                                "normalization": "imagenet",
                                "transform_config": {
                                    "train": None,
                                    "eval": None
                                },
                                "test_split_mode": "from_dir",
                                "test_split_ratio": 0.2,
                                "val_split_mode": "same_as_test",
                                "val_split_ratio": 0.5,
                                "tiling": {
                                    "apply": False,
                                    "tile_size": None,
                                    "stride": None,
                                    "remove_border_count": 0,
                                    "use_random_tiling": False,
                                    "random_tile_count": 16
                                }
                            },
                            "model": {
                                "name": "patchcore",
                                "model_artifact_name": "patchcore-model",
                                "export_path_root": "./artifacts",
                                "opset_version": 11,
                                "backbone": "wide_resnet50_2",
                                "pre_trained": True,
                                "layers": ["layer2", "layer3"],
                                "coreset_sampling_ratio": 0.05,
                                "num_neighbors": 10,
                                "normalization_method": "min_max"
                            },
                            "metrics": {
                                "image": ["F1Score", "AUROC"],
                                "pixel": ["F1Score", "AUROC"],
                                "threshold": {
                                    "method": "adaptive",
                                    "manual_image": None,
                                    "manual_pixel": None
                                }
                            },
                            "visualization": {
                                "show_images": False,
                                "save_images": True,
                                "log_images": True,
                                "image_save_path": None,
                                "mode": "full"
                            },
                            "project": {
                                "seed": 0,
                                "results_artifact_name": "patchcore-results",
                                "path": "./results"
                            },
                            "logging": {
                                "logger": [],
                                "log_graph": False
                            },
                            # "optimization": {
                            #     "export_mode": "onnx"
                            # },
                            "trainer": {
                                "enable_checkpointing": True,
                                "default_root_dir": None,
                                "gradient_clip_val": 0,
                                "gradient_clip_algorithm": "norm",
                                "num_nodes": 1,
                                "devices": 1,
                                "enable_progress_bar": True,
                                "overfit_batches": 0.0,
                                "track_grad_norm": -1,
                                "check_val_every_n_epoch": 1,
                                "fast_dev_run": False,
                                "accumulate_grad_batches": 1,
                                "max_epochs": 1,
                                "min_epochs": None,
                                "max_steps": -1,
                                "min_steps": None,
                                "max_time": None,
                                "limit_train_batches": 1.0,
                                "limit_val_batches": 1.0,
                                "limit_test_batches": 1.0,
                                "limit_predict_batches": 1.0,
                                "val_check_interval": 1.0,
                                "log_every_n_steps": 1,
                                "accelerator": "auto",
                                "strategy": None,
                                "sync_batchnorm": False,
                                "precision": 32,
                                "enable_model_summary": True,
                                "num_sanity_val_steps": 0,
                                "profiler": None,
                                "benchmark": False,
                                "deterministic": False,
                                "reload_dataloaders_every_n_epochs": 0,
                            }}
    )

    run.name = wandb.config["run_name"]
    art = wandb.use_artifact(wandb.config["dataset"]["dataset-artifact"])
    art.download(root=wandb.config["dataset"]["root"])
    
    configure_logger(level=wandb.config.log_level)

    if wandb.config.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    wandb_conf = OmegaConf.create(dict(wandb.config))
    with open("config.yml", "w+") as fp:
        OmegaConf.save(config=wandb_conf, f=fp.name)
        config = get_configurable_parameters(model_name=wandb.config["model"]["name"], config_path="config.yml")

    if config["project"]["seed"] is not None:
        seed_everything(config["project"]["seed"])

    datamodule = get_datamodule(config)
    model = get_model(config)

    # Add WandbLogger to log metrics to wandb
    wandb_logger = WandbLogger(log_model=True, 
                               save_code=True, 
                               checkpoint_name=f"{wandb.config['model']['model_artifact_name']}-torch", )
    
    experiment_logger = get_experiment_logger(config)
    experiment_logger.append(wandb_logger)

    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, 
                      logger=experiment_logger, 
                      callbacks=callbacks)
    
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)

    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member

    if config.dataset.test_split_mode == TestSplitMode.NONE:
        logger.info("No test set provided. Skipping test stage.")
    else:
        logger.info("Testing the model.")
        trainer.test(model=model, datamodule=datamodule)

    results_path = f"{wandb.config['project']['path']}/{wandb.config['model']['name']}/{wandb.config['dataset']['name']}"

    # Log model as W&B Artifact
    export_path = Path(f"{wandb.config['model']['export_path_root']}/{wandb.run.id}")
    export_path.mkdir(parents=True, exist_ok=True)
    onnx_path = export_path / "model.onnx"

    torch.onnx.export(
        model.model,
        torch.zeros((1, 3, wandb.config['dataset']["center_crop"], wandb.config['dataset']["center_crop"])).to(model.device),
        onnx_path,
        opset_version=wandb.config['model']["opset_version"],
        input_names=["input"],
        output_names=["output"],
    )

    # with open(f'{results_path}/run/onnx/meta_data.json', 'r') as json_file:
    #     metadata = json.load(json_file)
    model_art = wandb.Artifact(f"{wandb.config['model']['model_artifact_name']}-onnx", type="model")
    model_art.add_file(onnx_path)
    wandb.log_artifact(model_art)

    # Log results as W&B Artifact
    results_art = wandb.Artifact(wandb.config["project"]["results_artifact_name"], type="validation_results")
    results_art.add_dir(results_path + "/run/images")
    wandb.log_artifact(results_art)

    wandb.finish()

if __name__ == "__main__":
    train()