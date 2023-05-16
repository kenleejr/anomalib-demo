import json
import subprocess

def launch_job(job_name, entity, project, queue, config_path):
    cmd = f'wandb launch -j {job_name} -e {entity} -p {project} -q {queue} -c {config_path}'
    process = subprocess.Popen(cmd, shell=True)
    output, error = process.communicate()

    if error:
        print(f"Error occurred: {error}")
    else:
        print(f"Command executed successfully! Output: {output}")

def make_config(task_name):
    with open("./launch_jobs/anomalib-train/launch_configs/base_config.json", "r") as f:
        train_config = json.load(f)
    train_config["overrides"]["run_config"]["run_name"] = task_name + "-train"
    train_config["overrides"]["run_config"]["dataset"]["name"] = task_name
    train_config["overrides"]["run_config"]["dataset"]["dataset-artifact"] = task_name + ":latest"
    train_config["overrides"]["run_config"]["model"]["model_artifact_name"] = task_name + "-patchcore"
    train_config["overrides"]["run_config"]["model"]["registered_model_name"] = task_name
    with open(f"./launch_jobs/anomalib-train/launch_configs/{task_name}-train.json", "w+") as f:
        json.dump(train_config, f)

if __name__ == "__main__":
    task_names = ["bottle", "carpet",  "leather", "pill", "tile", "wood", "cable", "grid", "toothbrush" , "zipper", "capsule", "hazelnut" , "metal_nut", "screw", "transistor"]
    task_names = ["MVTec-" + x for x in task_names]
    for t in task_names:
        make_config(t)
    for t in task_names:
        launch_job(job_name="cvproject-trial-team/anomalib-demo/job-kenleejr_anomalib_train:latest", 
                   entity="cvproject-trial-team", 
                   project="anomalib-demo", 
                   queue="ec2-p3.2xlarge", 
                   config_path=f"./launch_jobs/anomalib-train/launch_configs/{t}-train.json")
