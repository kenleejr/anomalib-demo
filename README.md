# W&B Launch for Anomalib

This repo uses W&B Launch to facilitate scaled out re-training, evaluation, and deployment of `anomalib` anomaly detection models. 

[W&B Jobs](https://docs.wandb.ai/guides/launch/create-job) are configurable W&B runs which can be executed in any compute environment of your choice. 
W&B takes care of all the dependency management, containerization, versioning and results tracking of a job. 

The `launch_jobs` directory contains the scripts and Dockerfiles required to create jobs for common tasks such as training and evaluation of anomalib models. 
Currently there are two jobs supported: anomalib training and anomalib inference. 
These jobs take the [CLI utilities](https://github.com/openvinotoolkit/anomalib/blob/v0.4.0/tools/train.py) in the anomalib repo and turn them into portable W&B jobs that can be run on other infrastructure without having to worry about environment configuration. 

## Anomalib Training Job
In `launch_jobs/anomalib-train` there is everything you need to create a launch job: 
- `Dockerfile.train` installs dependencies for anomalib
- `train.py` containing training logic from anomalib
- `launch_jobs/anomalib-train/launch_configs/base_config.json` for the config to the launch job (dataset, model hyperparameters, etc.)

### Creating the Job in your W&B Project
1. First you must have a dataset of images logged as a W&B artifact, as this artifact will be an input to the job. Once you have that, copy and edit `base_config.json` to reflect this dataset artifact name and alias, hyperparameters, etc. you want for this intial job. 

```
## base_config.json
{"overrides": {
        "args": [],
        "run_config": {
        "log_level": "INFO",
        "show_images": true,
        "run_name": "MVTec-metal_nut-train", v.  ## Change this
        "dataset": {
            "name": "MVTec-metal_nut", ## Change this
            "root": "./artifacts/",
            "task": "segmentation",
            "format": "folder",
            "tiling": {
                "apply": false,
                "stride": null,
                "tile_size": null,
                "random_tile_count": 16,
                "use_random_tiling": false,
                "remove_border_count": 0
            },
            "mask_dir": "mask_dir/", 
            "extensions": ".png",
            "image_size": 256,
            "normal_dir": "normal_dir/",
            "center_crop": 224,
            "num_workers": 8,
            "abnormal_dir": "abnormal_dir/",
            "normalization": "imagenet",
            "val_split_mode": "same_as_test",
            "normal_test_dir": "normal_test_dir/",
            "test_batch_size": 32,
            "test_split_mode": "from_dir",
            "val_split_ratio": 0.5,
            "dataset-artifact": "MVTec-metal_nut:latest", ## Change this
            "test_split_ratio": 0.2,
            "train_batch_size": 32,
            "transform_config": {
                "eval": null,
                "train": null
            }
        },
        "model": {
            "name": "patchcore",
            "layers": [
                "layer2",
                "layer3"
            ],
            "backbone": "wide_resnet50_2",
            "pre_trained": true,
            "num_neighbors": 10,
            "export_path_root": "./artifacts",
            "onnx_opset_version": 11,
            "model_artifact_name": "MVTec-metal_nut-patchcore", ## Change this
            "registered_model_name": "MVTec-metal_nut", ## Change this
            "normalization_method": "min_max",
            "coreset_sampling_ratio": 0.05
```

3. Next [create a queue](https://docs.wandb.ai/guides/launch/create-queue) in your W&B team. For this example, just choose a `Docker` queue.
4. Run a [W&B agent](https://docs.wandb.ai/guides/launch/run-agent) in on a machine which has access to GPUs and has CUDA >= 11.7 and cuDNN >= 8:
```
pip install wandb
wandb login
wandb launch-agent -q <my_queue> -e <my_team> -j <num_parallel_jobs>
```
Now that machine is ready to receive training jobs and can execute `-j` number of jobs in parallel on that machine. It will poll the <queue> for jobs from W&B to execute.
  
4. The training container is already built and pushed to a public Dockerhub repo so you can use that out the box. From another terminal on your laptop or wherever you have `wandb` installed, run:
```
wandb launch -d kenleejr/anomalib:train -q <my_queue> -e <my_team> -p <my_project> -c <path_to_my_config.json>
```
This will launch the container (which just runs the anomalib training script with the config paramaters specified). Otherwise, you can build the container yourself and push to a repo of your choice:
```
docker build . -f launch_jobs/anomalib-train/Dockerfile.train -t <repo/image_name:tag>
docker push <repo/image_name:tag>
wandb launch -d <repo/image_name:tag> -q <my_queue> -e <my_team> -p <my_project> -c <path_to_my_config.json>
```
  
5. After the job is launched the first time, a named job will get created in your W&B project and can be re-used and re-configured however you like.
For instance, you may want to change out the training dataset or the hyperparameters of the training job that runs. Each time you run the above steps with a new image, W&B will create a new version of the job. To launch a specific job version, run:
```
wandb launch -j <my_job_name:alias> -q <my_queue> -e <my_entity> -p <my_project> -c <path_to_my_config.json>
```
or launch the job through the UI from the "Jobs" tab. 
