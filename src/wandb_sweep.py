# Import the W&B Python Library and log into W&B
import wandb
from train import main as train_main

# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score

project_name = "Prune"

# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "name": "dual_rm_sweep",
    "metric": {"goal": "minimize", "name": "charts/train_loss"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [1, 2, 3]},
        "lr": {"max": 0.1, "min": 0.0001},
        "use_cls": {"values": [True, False]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project=project_name)

wandb.agent(sweep_id, function=train_main(), count=2)