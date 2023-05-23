PROJECT_NAME = "anomalib-demo" 
ENTITY = "cvproject-trial-team"

import tritonclient.grpc as grpcclient
import numpy as np
from anomalib.data.utils import read_image, get_transforms
from anomalib.post_processing import (
    NormalizationMethod,
    ThresholdMethod,
    superimpose_anomaly_map,
)
from anomalib.pre_processing.transforms import Denormalize

import time
import pandas as pd
import wandb
import os 
import random
import datetime
import torch
import mon_sdk_dev
import signal

def log_wandb_table(inference_table, 
                    latest_version,
                    orig_image,
                    result,
                    score):
  
  i = orig_image
  r = result
  s = score
  # Convert the tensors to numpy arrays and wrap the image and masks with wandb.Image()
  anomaly_map = r.squeeze()
  orig_image = i.squeeze()
  heat_map = wandb.Image(superimpose_anomaly_map(anomaly_map=anomaly_map, image=Denormalize()(orig_image), normalize=True))

  # Extract other fields
  global_prediction = float(s.item())
  inference_table.add_data(datetime.datetime.now(), latest_version, heat_map, global_prediction)


def randomly_sample_image(directory: str):
  files = os.listdir(directory)
  # Filter out any non-image files
  images = [file for file in files if file.lower().endswith(('.png'))]
  # Randomly select an image
  image = random.choice(images)
  # Return the full path to the image
  return os.path.join(directory, image)

def send_triton_request(wandb_table,
                        image_path="./artifacts/MVTec-bottle:latest/abnormal_dir/broken_large_000.png",
                        model_name="MVTec-bottle"):
  client = grpcclient.InferenceServerClient(url="localhost:8001")

  transforms = get_transforms(image_size=256, center_crop=224, to_tensor=False)
  image = read_image(image_path).astype("float32")
  preprocessed_image = transforms(image=image)['image']
  image_data = np.expand_dims(preprocessed_image, axis=0).transpose(0, 3, 1, 2)

  input_tensors = [grpcclient.InferInput("input", image_data.shape, "FP32")]
  input_tensors[0].set_data_from_numpy(image_data)
  results = client.infer(model_name=model_name, inputs=input_tensors)

  model_repository_index = client.get_model_repository_index()
  print(type(model_repository_index))

  # Get most recent version of the model
  versions = []
  for model in model_repository_index.models:
      if model.name == model_name:
          versions.append(int(model.version))

  if versions:
      latest_version = max(versions)

  anomaly_map = results.as_numpy("output")
  anomaly_score = results.as_numpy("560")
  print("logging predictions")
  log_wandb_table(wandb_table, 
                  latest_version,
                  torch.tensor(image_data.squeeze()), 
                  anomaly_map.squeeze(), 
                  anomaly_score)


if __name__ == "__main__":
  def interrupt_handler(signum, frame):
    print("Interrupt received, stopping...")
    inference_table.join()
    exit(0)

  signal.signal(signal.SIGINT, interrupt_handler)
  task_names = ["bottle", "carpet",  "leather"]
  model_reg_names = ["MVTec-" + x for x in task_names]
  wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type="production-inference")
  inf_tables = {}
  for model_name in model_reg_names:
    inference_table = mon_sdk_dev.StreamTable(f"{model_name}-monitoring", ["timestamp",
                                                                          "model_version",
                                                                          "prediction_heatmap",
                                                                          "global_prediction"])
    inf_tables[f"{model_name}_inf_table"] = inference_table

  while(True):
    for model_name in model_reg_names:
      image_path = randomly_sample_image(f"./artifacts/{model_name}:latest/abnormal_dir")
      send_triton_request(inf_tables[f"{model_name}_inf_table"], image_path, model_name=model_name)
      time.sleep(1)
    time.sleep(5)