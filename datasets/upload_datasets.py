import wandb
from anomalib.data import get_datamodule
from anomalib.config import get_configurable_parameters

import os
import shutil

def move_images(src_dir, dst_dir, add_prefix=False, remove_suffix=False):
    for root, _, files in os.walk(src_dir):
        subdir_name = os.path.basename(root)
        for file in files:
            if file.endswith(".png"):
                src_file_path = os.path.join(root, file)
                new_file = file
                
                if add_prefix:
                    new_file = f"{subdir_name}_{new_file}"
                if remove_suffix:
                    new_file = new_file.replace("_mask", "")
                
                dst_file_path = os.path.join(dst_dir, new_file)
                shutil.copy(src_file_path, dst_file_path)

def download_mvtec(category: str = "bottle"):
    config = get_configurable_parameters(model_name="patchcore", config_path="datasets/config.yml")
    config["dataset"][category] = "bottle"
    datamodule = get_datamodule(config)
    datamodule.prepare_data()
    return datamodule

def convert_dataset_structure_to_generic_folder_format(source_dir="datasets/MVTec", dest_dir="datasets/MVTec-new", category: str = "bottle"):
    # Create destination directories
    os.makedirs(f"{dest_dir}/{category}/mask_dir", exist_ok=True)
    os.makedirs(f"{dest_dir}/{category}/abnormal_dir", exist_ok=True)
    os.makedirs(f"{dest_dir}/{category}/normal_test_dir", exist_ok=True)
    os.makedirs(f"{dest_dir}/{category}/normal_dir", exist_ok=True)

    # Move images from source to destination directories
    move_images(f"{source_dir}/{category}/ground_truth", f"{dest_dir}/{category}/mask_dir", add_prefix=True, remove_suffix=True)

    for subdir in os.listdir(f"{source_dir}/{category}/test"):
        src_path = os.path.join(f"{source_dir}/{category}/test", subdir)
        if os.path.isdir(src_path):
            if subdir == "good":
                move_images(src_path, f"{dest_dir}/{category}/normal_test_dir")
            else:
                move_images(src_path, f"{dest_dir}/{category}/abnormal_dir", add_prefix=True)

    move_images(f"{source_dir}/{category}/train/good", f"{dest_dir}/{category}/normal_dir")


def log_dataset_as_artifact(root_dir: str = "datasets/MVTec-new", category: str = "bottle"):
    art = wandb.Artifact(f"MVTec-{category}", type="dataset")
    art.add_dir(f"{root_dir}/{category}")
    wandb.log_artifact(art)
    

if __name__ == "__main__":
    wandb.init(project="anomalib-demo", entity="cvproject-trial-team")
    download_mvtec()
    for i in ["bottle", "carpet",  "leather", "pill", "tile", "wood", "cable", "grid", "toothbrush" , "zipper", "capsule", "hazelnut" , "metal_nut", "screw", "transistor"]:
        convert_dataset_structure_to_generic_folder_format(category=i)
        log_dataset_as_artifact(category=i)
    wandb.finish()
