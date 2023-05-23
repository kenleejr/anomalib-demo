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
from wandbmon import monitor

def pre_process(image_path, model_name):
  return {"orig_image": wandb.Image(image_path), "model_name": model_name}

def post_process(results):
  i, r, s = results
  # Convert the tensors to numpy arrays and wrap the image and masks with wandb.Image()
  anomaly_map = r.squeeze()
  orig_image = i.squeeze()
  heat_map = wandb.Image(superimpose_anomaly_map(anomaly_map=anomaly_map, image=Denormalize()(orig_image), normalize=True))

  # Extract other fields
  global_prediction = float(s.item())
  return {"prediction_heatmap": heat_map, "global_prediction": global_prediction}


def randomly_sample_image(directory: str):
  files = os.listdir(directory)
  # Filter out any non-image files
  images = [file for file in files if file.lower().endswith(('.png'))]
  # Randomly select an image
  image = random.choice(images)
  # Return the full path to the image
  return os.path.join(directory, image)

@monitor(
    input_preprocessor=pre_process, 
    output_postprocessor=post_process,
    settings={"project": "anomalib-demo",
              "entity": "kenlee"}
)
def send_triton_request(image_path="./artifacts/MVTec-bottle:latest/abnormal_dir/broken_large_000.png",
                        model_name="bottle"):
  client = grpcclient.InferenceServerClient(url="localhost:8001")

  transforms = get_transforms(image_size=256, center_crop=224, to_tensor=False)
  image = read_image(image_path).astype("float32")
  preprocessed_image = transforms(image=image)['image']
  image_data = np.expand_dims(preprocessed_image, axis=0).transpose(0, 3, 1, 2)

  input_tensors = [grpcclient.InferInput("input", image_data.shape, "FP32")]
  input_tensors[0].set_data_from_numpy(image_data)
  results = client.infer(model_name=model_name, inputs=input_tensors)
  anomaly_map = results.as_numpy("output")
  anomaly_score = results.as_numpy("560")

  print("logging predictions")
  return torch.tensor(image_data.squeeze()), anomaly_map.squeeze(), anomaly_score

if __name__ == "__main__":
  model_name = "bottle"
  while(True):
    image_path = randomly_sample_image(f"./artifacts/MVTec-{model_name}:latest/abnormal_dir")
    send_triton_request(image_path, model_name=model_name)
    time.sleep(5)