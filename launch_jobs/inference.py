PROJECT_NAME = "anomalib" 
ENTITY = "wandb-smle"

from pathlib import Path

import wandb
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import onnxruntime
import onnx

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from pytorch_lightning.loggers import WandbLogger
from anomalib.utils.loggers import get_experiment_logger

from omegaconf import OmegaConf

from anomalib.post_processing import (
    NormalizationMethod,
    ThresholdMethod,
    superimpose_anomaly_map,
)
from anomalib.pre_processing.transforms import Denormalize
from anomalib.utils.callbacks import (
    ImageVisualizerCallback,
    MetricsConfigurationCallback,
    MetricVisualizerCallback,
    PostProcessingConfigurationCallback,
)

def log_wandb_table(results):
  inference_table = wandb.Table(columns = ["prediction_pixels",
                                           "prediction_heatmap",
                                           "global_prediction", 
                                           "global_label"])

  for r in results:
    # Convert the tensors to numpy arrays and wrap the image and masks with wandb.Image()
    image = r["image"].numpy().squeeze().transpose(1, 2, 0)
    anomaly_map = r["anomaly_maps"].numpy().squeeze()
    pred_mask = r["pred_masks"].numpy().squeeze()
    pred_boxes = r["pred_boxes"]

    heat_map = wandb.Image(superimpose_anomaly_map(anomaly_map=anomaly_map, image=Denormalize()(r["image"]), normalize=True))

    # Create prediction image with segmentation masks
    class_labels = {
        1: "anomaly"
    }
    mask_data = pred_mask.astype(int)
    prediction = wandb.Image(image, masks={
        "predictions": {
            "mask_data": mask_data,
            "class_labels": class_labels
        }
    })
    # Create prediction_boxes image with bounding boxes
    boxes = []
    for i, b in enumerate(pred_boxes):
      b = b.squeeze()
      boxes.append({
          "position": {
              "minX": float(b[0]),
              "maxX": float(b[2]),
              "minY": float(b[1]),
              "maxY": float(b[3])
          },
          "class_id": int(r["box_labels"][i].item()),
          "box_caption": "Anomaly",
          "scores": {
              "score": float(r["box_scores"][i].item())
          }
      })
    prediction_boxes = wandb.Image(image, boxes={"predictions": {"box_data": boxes}})

    # Extract other fields
    global_prediction = float(r["pred_scores"].item())
    global_label = bool(r["pred_labels"].item())
    inference_table.add_data(prediction, heat_map, global_prediction, global_label)

  # Log the table
  wandb.log({"output_table": inference_table})

def infer():
    """Run inference."""
    # Initialize wandb
    wandb.init(project=PROJECT_NAME, 
               entity=ENTITY,
               job_type='inference',
               config={"log_level": "INFO",
                       "model": "patchcore",
                        "show_images": True,
                        "model_checkpoint": "patchcore-model:latest",
                       "inference_dataset": "MVTec-bottle:latest",
                       "inference_dataset_path": "test_abnormal",
                       "output": "inference_results",
                       "artifacts_root": "./artifacts",
                       "visualization_mode": "simple",
                       "show": False,
                            "dataset": {
                                "name": "MVTec-bottle",
                                "format": "folder",
                                "dataset-artifact": "MVTec-bottle:latest",
                                "root": "./artifacts/",
                                "normal_dir": "normal/",
                                "abnormal_dir": "test_abnormal/",
                                "normal_test_dir": "test_normal/",
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
                                "backbone": "wide_resnet50_2",
                                "pre_trained": True,
                                "layers": ["layer2", "layer3"],
                                "coreset_sampling_ratio": 0.1,
                                "num_neighbors": 9,
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
                                "path": "./results"
                            },
                            "logging": {
                                "logger": [],
                                "log_graph": False
                            },
                            "optimization": {
                                "export_mode": "onnx"
                            },
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
                            }
                        })

    # Retrieve versioned artifacts for inference
    inf_art = wandb.use_artifact(wandb.config["inference_dataset"])
    inf_art.download(root=wandb.config["artifacts_root"])

    model_art = wandb.use_artifact(wandb.config["model_checkpoint"])
    model_art.download(root=wandb.config["artifacts_root"])

    wandb_conf = OmegaConf.create(dict(wandb.config))
    with open("config.yml", "w+") as fp:
        OmegaConf.save(config=wandb_conf, f=fp.name)
        config = get_configurable_parameters(model_name=wandb.config["model"], config_path="config.yml")

    # config.trainer.resume_from_checkpoint = str(Path(wandb.config["artifacts_root"]) / "model.onnx")
    config.visualization.show_images = wandb.config["show"]
    config.visualization.mode = wandb.config["visualization_mode"]
    if wandb.config["output"]:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = wandb.config["output"]
    else:
        config.visualization.save_images = False
    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)

    # wandb_logger = WandbLogger(log_model=True, save_code=True)
    # experiment_logger = get_experiment_logger(config)
    # experiment_logger.append(wandb_logger)

    # trainer = Trainer(callbacks=callbacks, logger=experiment_logger, **config.trainer)

    # get the transforms
    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = tuple(center_crop)
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(
        config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
    )

    # create the dataset
    dataset = InferenceDataset(str(Path(wandb.config["artifacts_root"]) / wandb.config["inference_dataset_path"]), image_size=tuple(config.dataset.image_size), transform=transform)
    dataloader = DataLoader(dataset)

    model_path = str(Path(wandb.config["artifacts_root"]) / "model.onnx")

    # generate predictions
    # Load the ONNX model
    onnx_model = onnx.load(model_path)
    
    # Check the model
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s" % e)
    else:
        print("The model is valid!")

    session = onnxruntime.InferenceSession(model_path)

    # Run ONNX Runtime inference
    results = []
    for batch in dataloader:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(input_name)
        print(output_name)
        print(batch["image"].shape)
        result = session.run([output_name ,"554"], {input_name: batch["image"].numpy()})
        print(result)
        results.append(result)
    results = np.vstack(results)
    print(results)

    log_wandb_table(results)
    return results

if __name__ == "__main__":
    results = infer()
