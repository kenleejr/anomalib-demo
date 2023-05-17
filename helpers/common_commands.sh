docker run -v ./artifacts:/anomalib/artifacts --env WANDB_API_KEY= --shm-size=2g --gpus all kenleejr/anomalib:latest
docker build . -f launch_jobs/Dockerfile -t kenleejr/anomalib:latest
rsync -rlptzv --progress --exclude=.git --exclude=artifacts/ --exclude=wandb/ . "ubuntu@:/home/ubuntu/anomalib-demo"