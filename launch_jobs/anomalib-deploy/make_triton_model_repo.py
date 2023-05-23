import wandb
import os
import click

PROJECT_NAME = "anomalib-demo" 
ENTITY = "cvproject-trial-team"

@click.command()
@click.argument('version_index')
def create_local_model_repo(version_index):
    wandb.init(project=PROJECT_NAME, entity=ENTITY)
    # task_names = ["bottle", "carpet",  "leather", "pill", "tile", "wood", "cable", "grid", "toothbrush" , "zipper", "capsule", "hazelnut" , "metal_nut", "screw", "transistor"]
    task_names = ["bottle", "carpet",  "leather"]
    model_reg_names = ["MVTec-" + x for x in task_names]
    model_repo_root = "./artifacts/triton-model-repo"
    for m in model_reg_names:
        model_art = wandb.use_artifact(f"model-registry/{m}:latest")
        model_path = model_art.download(root=model_repo_root + f"/{m}/{version_index}/model.onnx")
    wandb.finish()

if __name__ == "__main__":
    create_local_model_repo()