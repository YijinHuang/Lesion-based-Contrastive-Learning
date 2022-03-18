import os
import pickle
import warnings

# import apex
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from modules import ContrastiveModel
from data import generate_dataset_from_pickle, DatasetFromDict, data_transforms


def generate_dataset(data_config, data_path, data_index):
    transform = data_transforms(data_config)
    datasets = generate_dataset_from_pickle(data_path, data_index, data_config, transform)

    return datasets


def generate_model(network, net_config, device, pretrained=True, checkpoint=None):
    if network not in net_config.keys():
        raise NotImplementedError('Not implemented network.')

    model = ContrastiveModel(
        net_config[network],
        pretrained,
        head='mlp',
        dim_in=2048,
        feat_dim=128
    ).to(device)

    if checkpoint:
        pretrained_model = torch.load(checkpoint).to(device)
        model.load_state_dict(pretrained_model.module.state_dict(), strict=True)
        print('Load weights form {}'.format(checkpoint))

    if device == 'cuda' and torch.cuda.device_count() > 1:
        # model = apex.parallel.convert_syncbn_model(model)
        warnings.warn('Data parallel is used for multiple GPUs training. \
            If you would like to apply synchronous batch normalization, please install apex and use apex.parallel.convert_syncbn_model.')
        model = torch.nn.DataParallel(model)

    return model


def auto_statistics(data_path, data_index, batch_size, num_workers, input_size):
    print('Calculating mean and std of training set for data normalization.')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    if data_index:
        train_set = pickle.load(open(data_index, 'rb'))['train']
        train_dataset = DatasetFromDict(train_set, transform=transform)
    else:
        train_path = os.path.join(data_path, 'train')
        train_dataset = datasets.ImageFolder(train_path, transform=transform)

    return mean_and_std(train_dataset, batch_size, num_workers)


def mean_and_std(train_dataset, batch_size, num_workers):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    num_samples = 0.
    channel_mean = torch.Tensor([0., 0., 0.])
    channel_std = torch.Tensor([0., 0., 0.])
    for samples in tqdm(loader):
        X, _ = samples
        channel_mean += X.mean((2, 3)).sum(0)
        num_samples += X.size(0)
    channel_mean /= num_samples

    for samples in tqdm(loader):
        X, _ = samples
        batch_samples = X.size(0)
        X = X.permute(0, 2, 3, 1).reshape(-1, 3)
        channel_std += ((X - channel_mean) ** 2).mean(0) * batch_samples
    channel_std = torch.sqrt(channel_std / num_samples)

    return channel_mean.tolist(), channel_std.tolist()


def print_msg(msg, appendixs=[]):
    max_len = len(max([msg, *appendixs], key=len))
    print('=' * max_len)
    print(msg)
    for appendix in appendixs:
        print(appendix)
    print('=' * max_len)


def show_config(configs):
    for name, config in configs.items():
        print('====={}====='.format(name))
        print_config(config)
        print('=' * (len(name) + 10))
        print()


def print_config(config, indentation=''):
    for key, value in config.items():
        if isinstance(value, dict):
            print('{}{}:'.format(indentation, key))
            print_config(value, indentation + '    ')
        else:
            print('{}{}: {}'.format(indentation, key, value))


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
