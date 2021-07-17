import os
import random
import shutil

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from config import *
from train import train
from utils import generate_dataset, generate_model, show_config


def main():
    # print configuration
    show_config({
        'BASIC CONFIG': BASIC_CONFIG,
        'DATA CONFIG': DATA_CONFIG,
        'TRAIN CONFIG': TRAIN_CONFIG
    })

    # reproducibility
    seed = BASIC_CONFIG['random_seed']
    set_random_seed(seed)

    # create folder
    save_path = BASIC_CONFIG['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # build model
    network = BASIC_CONFIG['network']
    device = BASIC_CONFIG['device']
    model = generate_model(
        network,
        NET_CONFIG,
        device,
        BASIC_CONFIG['pretrained'],
        BASIC_CONFIG['checkpoint']
    )

    # create dataset
    train_dataset, val_dataset = generate_dataset(
        DATA_CONFIG,
        BASIC_CONFIG['data_path'],
        BASIC_CONFIG['data_index']
    )

    # create logger
    record_path = BASIC_CONFIG['record_path']
    if os.path.exists(record_path):
        shutil.rmtree(record_path)
    logger = SummaryWriter(BASIC_CONFIG['record_path'])

    # create estimator and then train
    train(
        model=model,
        train_config=TRAIN_CONFIG,
        data_config=DATA_CONFIG,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        save_path=save_path,
        device=device,
        logger=logger
    )


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
