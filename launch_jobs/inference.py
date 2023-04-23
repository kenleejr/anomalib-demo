import wandb

if __name__ == "__main__":
    wandb.init(project="anomalib", entity='wandb-smle')
    wandb.log({"my_metric": 5})
    wandb.finish()