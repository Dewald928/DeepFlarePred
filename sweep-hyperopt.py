import wandb
from wandb.sweeps.config import tune
from wandb.sweeps.config.tune.suggest.hyperopt import HyperOptSearch
from wandb.sweeps.config.hyperopt import hp
import numpy as np

tune_config = tune.run(
    "main_TCN_Liu.py",
    search_alg=HyperOptSearch(
        dict(
            dropout=hp.uniform("dropout", 0.2, 1),
            learning_rate=hp.choice('learning_rate', [0.01,0.05,0.1,0.5,1])),
    metric="Validation_TSS",
    mode="max"),
    num_samples=100)

# Save sweep as yaml config file
tune_config.save("sweep-hyperopt.yaml")

# # Create the sweep
wandb.sweep(tune_config)
