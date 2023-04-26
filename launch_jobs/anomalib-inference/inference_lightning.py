"""Inference Entrypoint script."""
from pathlib import Path

import wandb
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

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
               config={"config_path": "src/anomalib/models/patchcore/config.yaml",
                       "weights": "results/patchcore/mvtec/bottle/run/weights/model.ckpt",
                       "input": "datasets/MVTec/bottle/test/contamination",
                       "output": "inference_results",
                       "visualization_mode": "simple",
                       "show": False})


    config = get_configurable_parameters(config_path=wandb.config["config_path"])
    config.trainer.resume_from_checkpoint = str(wandb.config["weights"])
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

    wandb_logger = WandbLogger(log_model=True, save_code=True)
    experiment_logger = get_experiment_logger(config)
    experiment_logger.append(wandb_logger)

    trainer = Trainer(callbacks=callbacks, logger=experiment_logger, **config.trainer)

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
    dataset = InferenceDataset(wandb.config["input"], image_size=tuple(config.dataset.image_size), transform=transform)
    dataloader = DataLoader(dataset)

    # generate predictions
    results = trainer.predict(model=model, dataloaders=[dataloader])
    log_wandb_table(results)
    return results

if __name__ == "__main__":
    results = infer()
