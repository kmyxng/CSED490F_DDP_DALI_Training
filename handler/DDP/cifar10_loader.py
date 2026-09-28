from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

base_dir="/SSD/CIFAR"

classes = (
    'plane',
    'car',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck')

CIFAR_MEAN = (0.4913999, 0.48215866, 0.44653133)
CIFAR_STD = (0.24703476, 0.24348757, 0.26159027)

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


transform_train = \
    transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4913999, 0.48215866, 0.44653133),
            (0.24703476, 0.24348757, 0.26159027))
        ])

transform_test = \
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4913999, 0.48215866, 0.44653133),
            (0.24703476, 0.24348757, 0.26159027))
        ])

        


def get_DDP_loader(test_batch, train_batch, root=base_dir, valid_size=0, valid_batch=0,
               cutout=16, num_workers=0, download=True, random_seed=12345, shuffle=True):
               
    ''' Problem 5: Get DDP loader
    (./handler/DDP/cifar10_loader.py)
    Implement a DDP loader for CIFAR-10 dataset.
    DDP loader is far simliar to the original loader, but you need to implement the distributed sampler.
    First at all, refer the 'handler/DP/cifar10_loader.py', which is the original loader.

    For the simplicity, we provide the scaffold codes from the original loader.
    You need to fill the missing parts.
    '''

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Requires an initialized distributed process group")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Batch sizes are global (same as DP), so each process loads 1/world_size of a batch.
    assert train_batch % world_size == 0
    assert test_batch % world_size == 0
    assert valid_batch % world_size == 0

    train_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True,
            download=download, transform=transform_train)

    if valid_size > 0:
        train_dataset, valid_dataset = \
            torch.utils.data.random_split(train_dataset,
                                          [50000-valid_size, valid_size],
                                          generator=torch.Generator().manual_seed(random_seed))
        valid_dataset.transforms = transform_test

    test_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False,
            download=download, transform=transform_test)

    if train_batch > 0:
        if cutout > 0:
            transform_train.transforms.append(Cutout(cutout))
        # DistributedSampler gives each rank a disjoint 1/world_size split of the dataset.
        # It shuffles by itself, so DataLoader's shuffle must stay False.
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=shuffle, seed=random_seed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch // world_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        train_loader = None

    if valid_size > 0:
        assert valid_batch > 0, "validation set follows the batch size of test set, which is 0"
        valid_sampler = DistributedSampler(
            valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=valid_batch // world_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        valid_loader = None

    if test_batch > 0:
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch // world_size, sampler=test_sampler,
            num_workers=num_workers, pin_memory=True, drop_last=False)
    else:
        test_loader = None

    return test_loader, train_loader, valid_loader
