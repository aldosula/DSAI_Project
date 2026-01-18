# Set random seeds BEFORE any other imports to ensure reproducibility
import random
import numpy as np
random.seed(1234)
np.random.seed(1234)

import torch
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

# torch.use_deterministic_algorithms(True)  # Disabled - causes avgpool CUDA error
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Set to False for full reproducibility

import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(version_base="1.1", config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    if config.disable_cudnn:
        print("cudnn is disabled, the trianing may be slower")
        torch.backends.cudnn.enabled = False
    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
