"""Reproducibility utilities."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, use deterministic algorithms (may be slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    logger.info(f"Set seed={seed}, deterministic={deterministic}")
