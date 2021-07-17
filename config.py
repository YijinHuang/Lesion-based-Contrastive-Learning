import resnet


BASIC_CONFIG = {
    'network': 'resnet50',  # shoud be one name in NET_CONFIG below
    'data_path': '/path/to/your/data/folder',  # preprocessed dataset folder
    'data_index': '/path/to/your/predicted/result/file',  # pickle file with lesion predicted results
    'save_path': './checkpoints',
    'record_path': './log',
    'pretrained': True,  # load pretrained parameters in ImageNet
    'checkpoint': None,  # load other pretrained model
    'random_seed': 1,  # random seed for reproducibilty
    'device': 'cuda'  # 'cuda' / 'cpu'
}

DATA_CONFIG = {
    'input_size': 128,
    'patch_size': 128,
    'mean': [0.425753653049469, 0.29737451672554016, 0.21293757855892181],
    'std': [0.27670302987098694, 0.20240527391433716, 0.1686241775751114],
    'data_augmentation': {
        'brightness': 0.4,  # how much to jitter brightness
        'contrast': 0.4,  # How much to jitter contrast
        'saturation': 0.4,
        'hue': 0.1,
        'scale': (0.8, 1.2),  # range of size of the origin size cropped
        'ratio': (0.8, 1.2),  # range of aspect ratio of the origin aspect ratio cropped
        'degrees': (-180, 180),  # range of degrees to select from
        'translate': (0.2, 0.2)  # tuple of maximum absolute fraction for horizontal and vertical translations
    }
}

TRAIN_CONFIG = {
    'epochs': 1000,  # total training epochs
    'batch_size': 786,  # training batch size
    'optimizer': 'SGD',  # SGD / ADAM
    'criterion': 'SupCon',  # 'CE' / 'MSE', cross entropy or mean squared error. Generally, MSE is better than CE on kappa.
    'learning_rate': 0.02,  # initial learning rate
    'lr_scheduler': 'cosine',  # one str name in SCHEDULER_CONFIG below, scheduler configurations are in SCHEDULER_CONFIG.
    'momentum': 0.9,  # momentum for SGD optimizer
    'nesterov': True,  # nesterov for SGD optimizer
    'weight_decay': 0.0005,  # weight decay for SGD and ADAM
    'kappa_prior': True,  # save model with higher kappa or higher accuracy in validation set
    'warmup_epochs': 0,  # warmup epochs
    'num_workers': 24,  # number of cpus used to load data at each step
    'save_interval': 50,  # number of epochs to store model
    'pin_memory': True,  # enables fast data transfer to CUDA-enabled GPUs
    'sample_view': True,  # visualize images on Tensorboard
    'sample_view_interval': 100,  # the steps interval of saving samples on Tensorboard. Note that frequently saving images will slow down the training speed.    
    # you can add any learning rate scheduler in torch.optim.lr_scheduler
    'scheduler_config': {
        'exponential': {
            'gamma': 0.9  # Multiplicative factor of learning rate decay
        },
        'multiple_steps': {
            'milestones': [150, 200],  # List of epoch indices. Must be increasing
            'gamma': 0.1,  # Multiplicative factor of learning rate decay
        },
        'cosine': {
            'T_max': 1000,  # Maximum number of iterations.
            'eta_min': 0  # Minimum learning rate.
        },
        'reduce_on_plateau': {
            'mode': 'min',  # In min mode, lr will be reduced when the quantity monitored has stopped decreasing
            'factor': 0.1,  # Factor by which the learning rate will be reduced
            'patience': 5,  # Number of epochs with no improvement after which learning rate will be reduced.
            'threshold': 1e-4,  # Threshold for measuring the new optimum
            'eps': 1e-5,  # Minimal decay applied to lr
        }
    }
}

# you can add any networks in torchvision.models
NET_CONFIG = {
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50,
    'resnet101': resnet.resnet101,
    'resnet152': resnet.resnet152,
    'resnext50': resnet.resnext50_32x4d,
    'resnext101': resnet.resnext101_32x8d,
    'args': {}
}
