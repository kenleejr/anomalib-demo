"""Inference Entrypoint script."""
PROJECT_NAME = "anomalib-demo" 
ENTITY = "cvproject-trial-team"

from pathlib import Path

import wandb
import numpy as np
from torch.utils.data import DataLoader
import torch
import onnxruntime
import onnx
import os

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

def log_wandb_table(model_name, results):
  inference_table = wandb.Table(columns = ["prediction_pixels",
                                           "prediction_heatmap",
                                           "global_prediction", 
                                           "global_label"])

  for r in results:
    # Convert the tensors to numpy arrays and wrap the image and masks with wandb.Image()
    image = r.transpose(1, 2, 0)
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
    # boxes = []
    # for i, b in enumerate(pred_boxes):
    #   if b.shape[0] != 4:
    #      continue
    #   b = b.squeeze()
    #   boxes.append({
    #       "position": {
    #           "minX": float(b[0]),
    #           "maxX": float(b[2]),
    #           "minY": float(b[1]),
    #           "maxY": float(b[3])
    #       },
    #       "class_id": int(r["box_labels"][i].item()),
    #       "box_caption": "Anomaly",
    #       "scores": {
    #           "score": float(r["box_scores"][i].item())
    #       }
    #   })
    # prediction_boxes = wandb.Image(image, boxes={"predictions": {"box_data": boxes}})

    # Extract other fields
    global_prediction = float(r["pred_scores"].item())
    global_label = bool(r["pred_labels"].item())
    inference_table.add_data(prediction, heat_map, global_prediction, global_label)

  # Log the table
  wandb.log({f"{model_name}/validation_table": inference_table})

def infer():
    """Run inference."""
    # Initialize wandb
    wandb.init(project=PROJECT_NAME, 
               entity=ENTITY,
               job_type='inference',
               config={"run_name": "MVTec-transistor-eval",
                       "log_level": "INFO",
                       "model": "patchcore",
                        "show_images": True,
                        "registered_model_name_alias": "MVTec-transistor:latest",
                       "inference_dataset": "MVTec-transistor:latest",
                       "inference_dataset_path": "abnormal_dir/",
                       "output": "inference_results",
                       "artifacts_root": "./artifacts",
                       "visualization_mode": "simple",
                       "show": False,
                            "dataset": {
                                "name": "MVTec-transistor",
                                "format": "folder",
                                "dataset-artifact": "MVTec-transistor:latest",
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
                                "model_artifact_name": "MVTec-transistor-patchcore",
                                "export_path_root": "./artifacts",
                                "onnx_opset_version": 11,
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

    wandb.config["trainer"]["limit_train_batches"] =  1.0
    wandb.config["trainer"]["limit_val_batches"] = 1.0
    wandb.config["trainer"]["limit_test_batches"] = 1.0
    wandb.config["trainer"]["limit_predict_batches"] = 1.0
    wandb.run.name = wandb.config["run_name"]
    
    # Retrieve versioned artifacts for inference
    inf_art = wandb.use_artifact(wandb.config["inference_dataset"])
    inf_path_at = inf_art.download()

    # model_art = wandb.use_artifact(f"model-registry/{wandb.config['registered_model_name_alias']}")
    model_art = wandb.use_artifact(f"{wandb.config['model']['model_artifact_name']}-onnx:latest")
    model_path_at = model_art.download()
    
    wandb_conf = OmegaConf.create(dict(wandb.config))
    with open("config.yml", "w+") as fp:
        OmegaConf.save(config=wandb_conf, f=fp.name)
        config = get_configurable_parameters(model_name=wandb.config["model"], config_path="config.yml")

    config.visualization.show_images = wandb.config["show"]
    config.visualization.mode = wandb.config["visualization_mode"]

    if wandb.config["output"]:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = wandb.config["output"]
    else:
        config.visualization.save_images = False

    wandb_logger = WandbLogger(log_model=True, save_code=True)
    experiment_logger = get_experiment_logger(config)
    experiment_logger.append(wandb_logger)

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
    inference_dataset_path = os.path.join(inf_path_at, wandb.config["inference_dataset_path"])
    print(inference_dataset_path)
    dataset = InferenceDataset(inference_dataset_path, 
                               image_size=tuple(config.dataset.image_size), 
                               transform=transform)
    print(dataset.__len__())
    dataloader = DataLoader(dataset)

    # generate predictions
    # Load the ONNX model
    model_path_at = model_path_at + "/model.onnx"
    print(model_path_at)
    onnx_model = onnx.load(model_path_at)
    
    # Check the model
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print("The model is invalid: %s" % e)
    else:
        print("The model is valid!")

    session = onnxruntime.InferenceSession(model_path_at, providers=['CUDAExecutionProvider'])

    # Run ONNX Runtime inference
    results = []
    for batch in dataloader:
        binding = session.io_binding()

        input_tensor = batch["image"].to('cuda:0').contiguous()

        binding.bind_input(
            name='input',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(input_tensor.shape),
            buffer_ptr=input_tensor.data_ptr(),
            )

        ## Allocate the PyTorch tensor for the model output
        Y_shape = (1, 1, wandb.config['dataset']['center_crop'], wandb.config['dataset']['center_crop']) 
        Y_tensor = torch.empty(Y_shape, dtype=torch.float32, device='cuda:0').to('cuda:0').contiguous()
        binding.bind_output(
            name='output',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=Y_shape,
            buffer_ptr=Y_tensor.data_ptr(),
        )

        session.run_with_iobinding(binding)
        results.append(Y_tensor)

    results = torch.vstack(results)
    results = results.to('cpu').numpy()
    print(results.shape)

    # log_wandb_table(wandb.config["registered_model_name_alias"].split(":")[0] + "-eval", results)
    return results

if __name__ == "__main__":
    results = infer()

    