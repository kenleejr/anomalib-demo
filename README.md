# W&B Anomalib Launch Demo

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
2. Next create a [create a queue](https://docs.wandb.ai/guides/launch/create-queue) in your W&B team. For this example, just choose a `Docker` queue.
3. Run a [W&B agent](https://docs.wandb.ai/guides/launch/run-agent) in on a machine which has access to GPUs:
```
pip install wandb
wandb login
wandb launch-agent -q <my_queue> -e <my_team> -j <num_parallel_jobs>
```
Now that machine is ready to receive training jobs and can execute `-j` number of jobs in parallel on that machine. It will poll the <queue> for jobs from W&B to execute. 
4. Wherever you have this repo cloned, run:
```
wandb launch -d kenleejr/anomalib:train -q <my_queue> -e <my_team> -p <my_project> -c <path_to_my_config.json>
```
This will launch the container (which just runs the anomalib training script with the config paramaters specified)
5. After the job is launched the first time, a named job will get created in your W&B project and can be re-used and re-configured however you like.
For instance, you may want to change out the training dataset or the hyperparameters of the training job that runs. Each time you run the above steps with a new image, W&B will create a new version of the job. To launch a specific job version, run:
```
wandb launch -j <my_job_name:alias> -q <my_queue> -e <my_entity> -p <my_project> -c <path_to_my_config.json>
```
