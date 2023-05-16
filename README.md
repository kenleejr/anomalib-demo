## W&B Anomalib Launch Demo

This repo uses W&B Launch to facilitate scaled out re-training, evaluation, and deployment of `anomalib` anomaly detection models. 

[W&B Jobs](https://docs.wandb.ai/guides/launch/create-job) are configurable W&B runs which can be executed in any compute environment of your choice. 
W&B takes care of all the dependency management, containerization, versioning and results tracking of a job. 

The `launch_jobs` directory contains the scripts and Dockerfiles required to create jobs for common tasks such as training and evaluation of anomalib models. 
Currently there are two jobs supported: anomalib training and anomalib inference. 
These jobs take the [CLI utilities](https://github.com/openvinotoolkit/anomalib/blob/v0.4.0/tools/train.py) in the anomalib repo and turn them into portable W&B jobs that can be run on other infrastructure without having to worry about environment configuration. 

To run jobs you will first need a to 1) [create a queue](https://docs.wandb.ai/guides/launch/create-queue) and 2) run a [W&B agent](https://docs.wandb.ai/guides/launch/run-agent) in infrastructure of your choice. For instance, if you want to run the anomalib training job on an EC2 instance, you would run:
```
pip install wandb
wandb launch-agent -q <my_queue> -e <my_entity> -j <num_parallel_jobs>
```
in a shell on the EC2 instance. To run an agent in a k8s cluster see the [Launch docs](https://docs.wandb.ai/guides/launch/kubernetes).

Once the agent is running, you are now ready to create and launch jobs to it. Once a job is created, you can launch it via CLI or the W&B UI. 
To create the training job run:
```
docker build . -f launch_jobs/anomalib-training/Dockerfile.train -t <my_image:tag>
docker push <my_image:tag>
```

To create the anomalib inference job:
```
docker build . -f launch_jobs/anomalib-inference/Dockerfile.inf -t <my_image:tag>
docker push <my_image:tag>
```
To launch the images, you can simply execute:
```
wandb launch -d <my_image:tag> -q <my_queue> -e <my_entity> -p <my_project> -c <path_to_my_config.json>
```
After the job is launched the first time, a named job will get created in your W&B project and can be re-used and re-configured however you like.
For instance, you may want to change out the training dataset or the hyperparameters of the training job that runs. Each time you run the above steps with a new image, W&B will create a new version of the job. To launch a specific job version, run:
```
wandb launch -j <my_job_name:alias> -q <my_queue> -e <my_entity> -p <my_project> -c <path_to_my_config.json>
```
