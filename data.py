import os
import pickle
import random

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def generate_dataset_from_pickle(data_path, pkl, data_config, transform):
    data = pickle.load(open(pkl, 'rb'))
    train_set, val_set = data['train'], data['val']

    train_dataset = LesionPatchGenerator(data_path, train_set, data_config['patch_size'], transform)
    val_dataset = LesionPatchGenerator(data_path, val_set, data_config['patch_size'], transform)

    return train_dataset, val_dataset


def data_transforms(data_config):
    data_aug = data_config['data_augmentation']
    input_size = data_config['input_size']
    mean, std = data_config['mean'], data_config['std']

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=data_aug['brightness'],
            contrast=data_aug['contrast'],
            saturation=data_aug['saturation'],
            hue=data_aug['hue']
        ),
        transforms.RandomResizedCrop(
            size=(input_size, input_size),
            scale=data_aug['scale'],
            ratio=data_aug['ratio']
        ),
        transforms.RandomAffine(
            degrees=data_aug['degrees'],
            translate=data_aug['translate']
        ),
        transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform


class LesionPatchGenerator(Dataset):
    def __init__(self, data_path, imgs, patch_size, transform=None):
        super(LesionPatchGenerator, self).__init__()
        self.imgs = []
        for name, lesions in imgs.items():
            path = os.path.join(data_path, name)
            self.imgs.append((path, lesions))
        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, lesions = self.imgs[index]
        img_path = img_path.replace('split_eyepacs_cropped', 'split_eyepacs_hq_512')
        img = self.pil_loader(img_path)

        bbox = random.choice(lesions)
        patch_1 = self.generate_patch(img, bbox)
        patch_2 = self.generate_patch(img, bbox)

        if self.transform is not None:
            patch_1 = self.transform(patch_1)
            patch_2 = self.transform(patch_2)

        return patch_1, patch_2

    def generate_patch(self, img, bbox):
        w, h = img.size
        x1, y1, x2, y2 = bbox
        # b_w = random.randint(32, 128)
        # b_h = random.randint(32, 128)
        # x1 = random.randint(10, w-b_w)
        # y1 = random.randint(10, h-b_h)
        # x2 = x1 + b_w
        # y2 = y1 + b_h
        b_w = bbox[2] - bbox[0]
        b_h = bbox[3] - bbox[1]

        x_space = self.patch_size - b_w
        if x1 < w - b_w:
            l_shift = int(random.random() * min(x1, x_space))
            new_x1 = x1 - l_shift
            new_x2 = x2 + (x_space - l_shift)
        else:
            r_shift = int(random.random() * min(w - b_w, x_space))
            new_x1 = x1 - (x_space - r_shift)
            new_x2 = x2 + r_shift

        y_space = self.patch_size - b_h
        if y1 < h - b_h:
            t_shift = int(random.random() * min(y1, y_space))
            new_y1 = y1 - t_shift
            new_y2 = y2 + (y_space - t_shift)
        else:
            d_shift = int(random.random() * min(h - b_h, y_space))
            new_y1 = y1 - (y_space - d_shift)
            new_y2 = y2 + d_shift

        patch = img.crop((new_x1, new_y1, new_x2, new_y2))
        return patch

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class DatasetFromDict(Dataset):
    def __init__(self, imgs, transform=None):
        super(DatasetFromDict, self).__init__()
        self.imgs = imgs
        self.transform = transform
        self.targets = [img[1] for img in imgs]
        self.classes = sorted(list(set(self.targets)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = self.pil_loader(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


class TwoCropTransform():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
