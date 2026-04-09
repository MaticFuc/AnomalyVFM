import os
import random
from enum import Enum

import numpy as np
import torch
import torch.optim as optim
from timm.optim import Muon


class OPTIMIZERS(Enum):
    ADAMW = "adamw"
    ADAM = "adam"
    MUON = "muon"

    def __str__(self):
        return self.value


class SCHEDULERS(Enum):
    NONE = "none"
    COS = "cos"
    MULTISTEPLR = "multisteplr"
    EXP = "exp"

    def __str__(self):
        return self.value


def get_optimizer(optimizer):
    if optimizer == OPTIMIZERS.ADAMW:
        return optim.AdamW
    elif optimizer == OPTIMIZERS.ADAM:
        return optim.Adam
    elif optimizer == OPTIMIZERS.MUON:
        return Muon
    else:
        raise Exception("Optimizer not supported")


def get_scheduler(scheduler, optimizer, **kwargs):
    if scheduler == SCHEDULERS.NONE:
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    elif scheduler == SCHEDULERS.COS:
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, kwargs["num_iterations"])
    elif scheduler == SCHEDULERS.MULTISTEPLR:
        return optim.lr_scheduler.MultiStepLR(
            optimizer,
            [int(0.8 * kwargs["num_iterations"]), int(0.9 * kwargs["num_iterations"])],
            gamma=0.5,
        )
    elif scheduler == SCHEDULERS.EXP:
        return optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    else:
        raise Exception("Scheduler not supported")


def torch_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
