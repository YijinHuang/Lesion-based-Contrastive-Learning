import os

import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules import *
from utils import print_msg
from utils import print_msg, inverse_normalize


def train(model, train_config, data_config, train_dataset, val_dataset, save_path, device, logger=None):
    optimizer = initialize_optimizer(train_config, model)
    lr_scheduler, warmup_scheduler = initialize_lr_scheduler(train_config, optimizer)
    loss_function = ContrastiveLoss().to(device)
    train_loader, val_loader = initialize_dataloader(train_config, train_dataset, val_dataset)

    # start training
    model.train()
    min_indicator = 9999999
    avg_loss = 0
    for epoch in range(1, train_config['epochs'] + 1):
        # warmup scheduler update
        if warmup_scheduler and not warmup_scheduler.is_finish():
            warmup_scheduler.step()

        epoch_loss = 0
        progress = tqdm(enumerate(train_loader))
        for step, train_data in progress:
            X_1, X_2 = train_data
            X = torch.cat([X_1, X_2], dim=0)
            X = X.to(device)
            bsz = X_1.shape[0]

            # forward
            features = model(X)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = loss_function(features)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            epoch_loss += loss.item()
            avg_loss = epoch_loss / (step + 1)

            progress.set_description(
                'epoch: {}, loss: {:.6f}'
                .format(epoch, avg_loss)
            )

            # visualize samples
            if train_config['sample_view'] and step % train_config['sample_view_interval'] == 0:
                samples = torchvision.utils.make_grid(X)
                samples = inverse_normalize(samples, data_config['mean'], data_config['std'])
                logger.add_image('input samples', samples, 0, dataformats='CHW')

        # validation performance
        if epoch % 10 == 0:
            val_loss = eval(model, val_loader, loss_function, device)
            logger.add_scalar('validation loss', val_loss, epoch)
            print('validation loss: {:.6f}'.format(val_loss))

            # save model
            indicator = val_loss
            if indicator < min_indicator:
                torch.save(model, os.path.join(save_path, 'best_validation_model.pt'))
                min_indicator = indicator
                print_msg('Best in validation set. Model save at {}'.format(save_path))

        if epoch % train_config['save_interval'] == 0:
            torch.save(model, os.path.join(save_path, 'epoch_{}.pt'.format(epoch)))

        # update learning rate
        if lr_scheduler and (not warmup_scheduler or warmup_scheduler.is_finish()):
            if train_config['lr_scheduler'] == 'reduce_on_plateau':
                lr_scheduler.step(avg_loss)
            else:
                lr_scheduler.step()

        # record
        if logger:
            logger.add_scalar('training loss', avg_loss, epoch)

    # save final model
    torch.save(model, os.path.join(save_path, 'final_model.pt'))
    if logger:
        logger.close()


def evaluate(model_path, train_config, test_dataset, num_classes, estimator, device):
    trained_model = torch.load(model_path).to(device)
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config['batch_size'],
        num_workers=train_config['num_workers'],
        shuffle=False
    )

    print('Running on Test set...')
    eval(trained_model, test_loader, train_config['criterion'], estimator, device)


def eval(model, dataloader, loss_function, device):
    model.eval()
    torch.set_grad_enabled(False)

    # estimator.reset()
    val_loss = 0
    avg_loss = 0
    for step, test_data in enumerate(dataloader):
        X_1, X_2 = test_data
        X = torch.cat([X_1, X_2], dim=0)
        X = X.to(device)
        bsz = X_1.shape[0]

        # forward
        features = model(X)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = loss_function(features)
        val_loss += loss.item()
        avg_loss = val_loss / (step + 1)

    model.train()
    torch.set_grad_enabled(True)
    return avg_loss


# define data loader
def initialize_dataloader(train_config, train_dataset, val_dataset):
    batch_size = train_config['batch_size']
    num_workers = train_config['num_workers']
    pin_memory = train_config['pin_memory']
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


# define optmizer
def initialize_optimizer(train_config, model):
    optimizer_strategy = train_config['optimizer']
    learning_rate = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    momentum = train_config['momentum']
    nesterov = train_config['nesterov']
    if optimizer_strategy == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay
        )
    elif optimizer_strategy == 'ADAM':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError('Not implemented optimizer.')

    return optimizer


# define learning rate scheduler
def initialize_lr_scheduler(train_config, optimizer):
    learning_rate = train_config['learning_rate']
    warmup_epochs = train_config['warmup_epochs']
    scheduler_strategy = train_config['lr_scheduler']
    scheduler_config = train_config['scheduler_config']

    lr_scheduler = None
    if scheduler_strategy in scheduler_config.keys():
        scheduler_config = scheduler_config[scheduler_strategy]
        if scheduler_strategy == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_config)
        elif scheduler_strategy == 'multiple_steps':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_config)
        elif scheduler_strategy == 'reduce_on_plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_config)
        elif scheduler_strategy == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_config)

    if warmup_epochs > 0:
        warmup_scheduler = WarmupLRScheduler(optimizer, warmup_epochs, learning_rate)
    else:
        warmup_scheduler = None

    return lr_scheduler, warmup_scheduler
