import numpy as np
import os
from PIL import Image
import pandas as pd

import torch
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
from torchvision import transforms, datasets

from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import utils.wilds_utils as wu


def get_in_distribution_data_loaders(args, device):
    """ load in-distribution datasets and return data loaders """
    num_features = None  # Default value

    if args.benchmark in ['R-MNIST', 'MNIST-OOD']:
        if args.benchmark == 'R-MNIST':
            no_loss_acc = False
            # here, id is the rotation angle
            ids = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        else:
            no_loss_acc = True
            # here, id is the name of the dataset
            ids = ['MNIST', 'EMNIST', 'FMNIST', 'KMNIST']
        train_loader, val_loader, in_test_loader = get_mnist_loaders(
            args.data_root,
            model_class=args.model,
            batch_size=args.batch_size,
            val_size=args.val_set_size,
            download=args.download,
            device=device)
        return (train_loader, val_loader, in_test_loader), ids, no_loss_acc, num_features

    elif args.benchmark in ['R-FMNIST', 'FMNIST-OOD']:
        if args.benchmark == 'R-FMNIST':
            no_loss_acc = False
            # here, id is the rotation angle
            ids = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
        else:
            no_loss_acc = True
            # here, id is the name of the dataset
            ids = ['FMNIST', 'EMNIST', 'MNIST', 'KMNIST']
        train_loader, val_loader, in_test_loader = get_fmnist_loaders(
            args.data_root,
            model_class=args.model,
            batch_size=args.batch_size,
            val_size=args.val_set_size,
            download=args.download,
            device=device)
        return (train_loader, val_loader, in_test_loader), ids, no_loss_acc, num_features

    elif args.benchmark in ['CIFAR-10-C', 'CIFAR-10-OOD']:
        if args.benchmark == 'CIFAR-10-C':
            no_loss_acc = False
            # here, id is the corruption severity
            ids = [0, 1, 2, 3, 4, 5]
        else:
            no_loss_acc = True
            # here, id is the name of the OOD dataset
            ids = ['CIFAR-10', 'SVHN', 'LSUN', 'CIFAR-100']

        train_loader, val_loader, in_test_loader = get_cifar10_loaders(
            args.data_root,
            batch_size=args.batch_size,
            train_batch_size=args.batch_size,
            val_size=args.val_set_size,
            download=args.download,
            data_augmentation=not args.noda)
        return (train_loader, val_loader, in_test_loader), ids, no_loss_acc, num_features

    elif args.benchmark == 'ImageNet-C':
        no_loss_acc = False
        # here, id is the corruption severity
        ids = [0, 1, 2, 3, 4, 5]
        train_loader, val_loader, in_test_loader = get_imagenet_loaders(
            args.data_root,
            batch_size=args.batch_size,
            train_batch_size=args.batch_size,
            val_size=args.val_set_size)
        return (train_loader, val_loader, in_test_loader), ids, no_loss_acc, num_features

    elif 'WILDS' in args.benchmark:
        dataset = args.benchmark[6:]
        no_loss_acc = False
        ids = [f'{dataset}-id', f'{dataset}-ood']
        train_loader, val_loader, in_test_loader = wu.get_wilds_loaders(
            dataset, args.data_root, args.data_fraction, args.model_seed)
        return (train_loader, val_loader, in_test_loader), ids, no_loss_acc, num_features

    elif args.benchmark == 'Adult':
        no_loss_acc = False
        # For the Adult dataset, the concept of OOD IDs is handled by domain_shift_gender and noise_intensity args
        # We can just use placeholder IDs.
        ids = ['Adult_Test']
        (train_loader, val_loader, in_test_loader), num_features = get_adult_loaders(
            args.data_root,
            batch_size=args.batch_size,
            domain_shift_gender=args.domain_shift_gender,
            noise_intensity=args.noise_intensity,
            device=device
        )
        return (train_loader, val_loader, in_test_loader), ids, no_loss_acc, num_features

    # This final return is a fallback, though the logic should always hit one of the branches above.
    return None, None, None, None

def get_ood_test_loader(args, id):
    """ load out-of-distribution test data and return data loader """

    if args.benchmark == 'R-MNIST':
        _, test_loader = get_rotated_mnist_loaders(
            id, args.data_root,
            model_class=args.model,
            download=args.download)
    elif args.benchmark == 'R-FMNIST':
        _, test_loader = get_rotated_fmnist_loaders(
            id, args.data_root,
            model_class=args.model,
            download=args.download)
    elif args.benchmark == 'CIFAR-10-C':
        test_loader = load_corrupted_cifar10(
            id, data_dir=args.data_root,
            batch_size=args.batch_size,
            cuda=torch.cuda.is_available())
    elif args.benchmark == 'ImageNet-C':
        test_loader = load_corrupted_imagenet(
            id, data_dir=args.data_root,
            batch_size=args.batch_size,
            cuda=torch.cuda.is_available())
    elif args.benchmark == 'MNIST-OOD':
        _, test_loader = get_mnist_ood_loaders(
            id, data_path=args.data_root,
            batch_size=args.batch_size,
            download=args.download)
    elif args.benchmark == 'FMNIST-OOD':
        _, test_loader = get_mnist_ood_loaders(
            id, data_path=args.data_root,
            batch_size=args.batch_size,
            download=args.download)
    elif args.benchmark == 'CIFAR-10-OOD':
        _, test_loader = get_cifar10_ood_loaders(
            id, data_path=args.data_root,
            batch_size=args.batch_size,
            download=args.download)
    elif 'WILDS' in args.benchmark:
        dataset = args.benchmark[6:]
        test_loader = wu.get_wilds_ood_test_loader(
            dataset, args.data_root, args.data_fraction)

    return test_loader


def val_test_split(dataset, val_size=5000, batch_size=512, num_workers=5, pin_memory=False):
    # Split into val and test sets
    test_size = len(dataset) - val_size
    dataset_val, dataset_test = data_utils.random_split(
        dataset, (val_size, test_size), generator=torch.Generator().manual_seed(42)
    )
    val_loader = data_utils.DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=pin_memory)
    test_loader = data_utils.DataLoader(dataset_test, batch_size=batch_size, shuffle=False,
                                        num_workers=num_workers, pin_memory=pin_memory)
    return val_loader, test_loader


def get_cifar10_loaders(data_path, batch_size=512, val_size=2000,
                        train_batch_size=128, download=False, data_augmentation=True):
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    tforms = [transforms.ToTensor(),
              transforms.Normalize(mean, std)]
    tforms_test = transforms.Compose(tforms)
    if data_augmentation:
        tforms_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4)]
                                        + tforms)
    else:
        tforms_train = tforms_test

    # Get datasets and data loaders
    train_set = datasets.CIFAR10(data_path, train=True, transform=tforms_train,
                                 download=download)
    # train_set = data_utils.Subset(train_set, range(500))
    val_test_set = datasets.CIFAR10(data_path, train=False, transform=tforms_test,
                                    download=download)

    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=train_batch_size,
                                         shuffle=True)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size)

    return train_loader, val_loader, test_loader


def get_imagenet_loaders(data_path, batch_size=128, val_size=2000,
                         train_batch_size=128, num_workers=5):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    tforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    tforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    data_path_train = os.path.join(data_path, 'ImageNet2012/train')
    data_path_val = os.path.join(data_path, 'ImageNet2012/val')

    # Get datasets and data loaders
    train_set = datasets.ImageFolder(data_path_train, transform=tforms_train)
    val_test_set = datasets.ImageFolder(data_path_val, transform=tforms_test)

    train_loader = data_utils.DataLoader(train_set,
                                         batch_size=train_batch_size,
                                         shuffle=True,
                                         num_workers=num_workers,
                                         pin_memory=True)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size,
                                             num_workers=num_workers,
                                             pin_memory=True)

    return train_loader, val_loader, test_loader


def get_mnist_loaders(data_path, batch_size=512, model_class='LeNet',
                      train_batch_size=128, val_size=2000, download=False, device='cpu'):
    if model_class == "MLP":
        tforms = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])
    else:
        tforms = transforms.ToTensor()

    train_set = datasets.MNIST(data_path, train=True, transform=tforms,
                               download=download)
    val_test_set = datasets.MNIST(data_path, train=False, transform=tforms,
                                  download=download)

    Xys = [train_set[i] for i in range(len(train_set))]
    Xs = torch.stack([e[0] for e in Xys]).to(device)
    ys = torch.stack([torch.tensor(e[1]) for e in Xys]).to(device)
    train_loader = FastTensorDataLoader(Xs, ys, batch_size=batch_size, shuffle=True)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size)

    return train_loader, val_loader, test_loader


def get_fmnist_loaders(data_path, batch_size=512, model_class='LeNet',
                       train_batch_size=128, val_size=2000, download=False, device='cpu'):
    if model_class == "MLP":
        tforms = transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))])
    else:
        tforms = transforms.ToTensor()

    train_set = datasets.FashionMNIST(data_path, train=True, transform=tforms,
                                      download=download)
    val_test_set = datasets.FashionMNIST(data_path, train=False, transform=tforms,
                                         download=download)

    Xys = [train_set[i] for i in range(len(train_set))]
    Xs = torch.stack([e[0] for e in Xys]).to(device)
    ys = torch.stack([torch.tensor(e[1]) for e in Xys]).to(device)
    train_loader = FastTensorDataLoader(Xs, ys, batch_size=batch_size, shuffle=True)
    val_loader, test_loader = val_test_split(val_test_set,
                                             batch_size=batch_size,
                                             val_size=val_size)

    return train_loader, val_loader, test_loader


def get_rotated_mnist_loaders(angle, data_path, model_class='LeNet', download=False):
    if model_class == "MLP":
        shift_tforms = transforms.Compose([RotationTransform(angle), transforms.ToTensor(),
                                           ReshapeTransform((-1,))])
    else:
        shift_tforms = transforms.Compose([RotationTransform(angle), transforms.ToTensor()])

    # Get rotated MNIST val/test sets and loaders
    rotated_mnist_val_test_set = datasets.MNIST(data_path, train=False,
                                                transform=shift_tforms,
                                                download=download)
    shift_val_loader, shift_test_loader = val_test_split(rotated_mnist_val_test_set,
                                                         val_size=2000)

    return shift_val_loader, shift_test_loader


def get_rotated_fmnist_loaders(angle, data_path, model_class='LeNet', download=False):
    if model_class == "MLP":
        shift_tforms = transforms.Compose([RotationTransform(angle), transforms.ToTensor(),
                                           ReshapeTransform((-1,))])
    else:
        shift_tforms = transforms.Compose([RotationTransform(angle), transforms.ToTensor()])

    # Get rotated FMNIST val/test sets and loaders
    rotated_fmnist_val_test_set = datasets.FashionMNIST(data_path, train=False,
                                                        transform=shift_tforms,
                                                        download=download)
    shift_val_loader, shift_test_loader = val_test_split(rotated_fmnist_val_test_set,
                                                         val_size=2000)

    return shift_val_loader, shift_test_loader


# https://discuss.pytorch.org/t/missing-reshape-in-torchvision/9452/6
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class RotationTransform:
    """Rotate the given angle."""
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


def uniform_noise(dataset, delta=1, size=5000, batch_size=512):
    if dataset in ['MNIST', 'FMNIST', 'R-MNIST']:
        shape = (1, 28, 28)
    elif dataset in ['SVHN', 'CIFAR10', 'CIFAR100', 'CIFAR-10-C']:
        shape = (3, 32, 32)
    elif dataset in ['ImageNet', 'ImageNet-C']:
        shape = (3, 256, 256)

    # data = torch.rand((100*batch_size,) + shape)
    data = delta * torch.rand((size,) + shape)
    train = data_utils.TensorDataset(data, torch.zeros_like(data))
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
    return loader


class DatafeedImage(torch.utils.data.Dataset):
    def __init__(self, x_train, y_train, transform=None):
        self.x_train = x_train
        self.y_train = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.x_train[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y_train[index]

    def __len__(self):
        return len(self.x_train)


def load_corrupted_cifar10(severity, data_dir='data', batch_size=256, cuda=True,
                           workers=0):
    """
    Load corrupted CIFAR-10 dataset for a specific severity level.
    This version correctly handles the data format from the Zenodo download.
    """
    if not isinstance(severity, int) or not (0 <= severity <= 5):
        raise ValueError("Severity must be an integer between 0 and 5.")

    # Severity 0 corresponds to the original, uncorrupted test set.
    if severity == 0:
        _, _, test_loader = get_cifar10_loaders(data_dir, batch_size)
        return test_loader

    corruption_types = [
        'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
        'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
        'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
        'snow', 'spatter', 'speckle_noise', 'zoom_blur'
    ]
    data_dir_c = os.path.join(data_dir, 'CIFAR-10-C')

    # Load all corruption types for the given severity.
    all_x = list()
    for corruption in corruption_types:
        x_file = os.path.join(data_dir_c, f"{corruption}.npy")
        if not os.path.exists(x_file):
            raise FileNotFoundError(f"Missing corruption file: {x_file}. Please ensure CIFAR-10-C is downloaded and extracted correctly.")

        data_chunk = np.load(x_file)
        # Each severity level has 10,000 images.
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        all_x.append(data_chunk[start_idx:end_idx])

    # Concatenate all corruption types for the given severity.
    np_x = np.concatenate(all_x)

    # Load labels and tile them for each corruption type.
    y_file = os.path.join(data_dir_c, 'labels.npy')
    if not os.path.exists(y_file):
        raise FileNotFoundError(f"Missing labels file: {y_file}. Please ensure CIFAR-10-C is downloaded and extracted correctly.")

    labels_chunk = np.load(y_file).astype(np.int64)
    labels_for_severity = labels_chunk[(severity - 1) * 10000 : severity * 10000]
    np_y = np.tile(labels_for_severity, len(corruption_types))

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = DatafeedImage(np_x, np_y, transform)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=cuda)

    return loader


def load_corrupted_imagenet(severity, data_dir='data', batch_size=128, cuda=True, workers=1):
    """ load corrupted ImageNet dataset """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
    ])

    corruption_types = ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog',
                        'frost', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise',
                        'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise',
                        'snow', 'spatter', 'speckle_noise', 'zoom_blur']

    dsets = list()
    for c in corruption_types:
        path = os.path.join(data_dir, 'ImageNet-C/' + c + '/' + str(severity))
        dsets.append(datasets.ImageFolder(path,
                                          transform=transform))
    dataset = data_utils.ConcatDataset(dsets)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=cuda)

    return loader


def get_mnist_ood_loaders(ood_dataset, data_path='./data', batch_size=512, download=False):
    '''Get out-of-distribution val/test sets and val/test loaders (in-distribution: MNIST/FMNIST)'''
    tforms = transforms.ToTensor()
    if ood_dataset == 'FMNIST':
        fmnist_val_test_set = datasets.FashionMNIST(data_path, train=False,
                                                    transform=tforms,
                                                    download=download)
        val_loader, test_loader = val_test_split(fmnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'EMNIST':
        emnist_val_test_set = datasets.EMNIST(data_path, split='digits', train=False,
                                              transform=tforms,
                                              download=download)
        val_loader, test_loader = val_test_split(emnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'KMNIST':
        kmnist_val_test_set = datasets.KMNIST(data_path, train=False,
                                              transform=tforms,
                                              download=download)
        val_loader, test_loader = val_test_split(kmnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'MNIST':
        mnist_val_test_set = datasets.MNIST(data_path, train=False,
                                            transform=tforms,
                                            download=download)
        val_loader, test_loader = val_test_split(mnist_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    else:
        raise ValueError('Choose one out of FMNIST, EMNIST, MNIST, and KMNIST.')
    return val_loader, test_loader


def get_cifar10_ood_loaders(ood_dataset, data_path='./data', batch_size=512, download=False):
    '''Get out-of-distribution val/test sets and val/test loaders (in-distribution: CIFAR-10)'''
    if ood_dataset == 'SVHN':
        svhn_tforms = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4376821, 0.4437697, 0.47280442),
                                                   (0.19803012, 0.20101562, 0.19703614))])
        svhn_val_test_set = datasets.SVHN(data_path, split='test',
                                          transform=svhn_tforms,
                                          download=download)
        val_loader, test_loader = val_test_split(svhn_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    elif ood_dataset == 'LSUN':
        lsun_tforms = transforms.Compose([transforms.Resize(size=(32, 32)),
                                          transforms.ToTensor()])
        lsun_test_set = datasets.LSUN(data_path, classes=['classroom_val'],  # classes='test'
                                      transform=lsun_tforms)
        val_loader = None
        test_loader = data_utils.DataLoader(lsun_test_set, batch_size=batch_size,
                                            shuffle=False)
    elif ood_dataset == 'CIFAR-100':
        cifar100_tforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                   (0.2023, 0.1994, 0.2010))])
        cifar100_val_test_set = datasets.CIFAR100(data_path, train=False,
                                                  transform=cifar100_tforms,
                                                  download=download)
        val_loader, test_loader = val_test_split(cifar100_val_test_set,
                                                 batch_size=batch_size,
                                                 val_size=0)
    else:
        raise ValueError('Choose one out of SVHN, LSUN, and CIFAR-100.')
    return val_loader, test_loader


class FastTensorDataLoader:
    """
    Source: https://github.com/hcarlens/pytorch-tabular/blob/master/fast_tensor_data_loader.py
    and https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(self, *tensors, batch_size=32, shuffle=False):
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors
        self.dataset = tensors[0]

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


class PermutedMnistGenerator():
    def __init__(self, data_path='./data', num_tasks=10, random_seed=0, download=False):
        self.data_path = data_path
        self.num_tasks = num_tasks
        self.random_seed = random_seed
        self.download = download
        self.out_dim = 10           # number of classes in the MNIST dataset
        self.in_dim = 784           # each image has 28x28 pixels
        self.task_id = 0            # initialize the current task id

    def next_task(self, batch_size=256, val_size=0):
        if self.task_id >= self.num_tasks:
            raise Exception('Number of tasks exceeded!')
        else:
            np.random.seed(self.task_id+self.random_seed)
            perm_inds = np.arange(self.in_dim)

            # First task is (unpermuted) MNIST, subsequent tasks are random permutations of pixels
            if self.task_id > 0:
                np.random.shuffle(perm_inds)

            # make image a tensor and permute pixel values
            tforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1)[perm_inds]),
            ])

            # load datasets
            train_set = datasets.MNIST(self.data_path, train=True,
                                       transform=tforms, download=self.download)
            val_test_set = datasets.MNIST(self.data_path, train=False,
                                          transform=tforms, download=self.download)

            # fast DataLoader for training
            Xys = [train_set[i] for i in range(len(train_set))]
            Xs = torch.stack([e[0] for e in Xys])
            ys = torch.stack([torch.tensor(e[1]) for e in Xys])
            train_loader = FastTensorDataLoader(Xs, ys, batch_size=batch_size, shuffle=True)
            val_loader, test_loader = val_test_split(val_test_set,
                                                     batch_size=batch_size,
                                                     val_size=val_size,
                                                     num_workers=0)

            # increment task counter
            self.task_id += 1

            if val_size > 0:
                return train_loader, val_loader, test_loader
            return train_loader, test_loader


def get_adult_loaders(data_path, batch_size=256, val_size=0.2, domain_shift_gender=None, noise_intensity=0.0,
                      device='cpu'):
    """
    Loads and preprocesses the Adult Income dataset.
    Handles standard splits, domain shift splits by gender, and adding noise to the test set.
    """
    col_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]

    # Load data
    train_df = pd.read_csv(os.path.join(data_path, 'adult.data'), header=None, names=col_names, na_values=' ?',
                           skipinitialspace=True)
    test_df = pd.read_csv(os.path.join(data_path, 'adult.test'), header=None, names=col_names, na_values=' ?',
                          skiprows=1, skipinitialspace=True)

    test_df['income'] = test_df['income'].str.replace(r'\.', '', regex=True)
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Preprocessing
    df.dropna(inplace=True)
    df = df.drop(columns=['fnlwgt', 'education'])
    df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)

    # Define feature types - FIX: 'sex' is now treated as a normal categorical feature
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=np.number).drop(columns=['income']).columns.tolist()

    # Create preprocessing pipeline - FIX: No special 'remainder' handling needed
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    # Splitting logic remains the same, works on the original dataframe
    if domain_shift_gender:
        if domain_shift_gender == 'male_to_female':
            train_data_unsplit = df[df['sex'] == 'Male']
            test_data = df[df['sex'] == 'Female']
        elif domain_shift_gender == 'female_to_male':
            train_data_unsplit = df[df['sex'] == 'Female']
            test_data = df[df['sex'] == 'Male']
        else:
            raise ValueError("domain_shift_gender must be 'male_to_female' or 'female_to_male'")

        train_data, val_data = sk_train_test_split(train_data_unsplit, test_size=val_size, random_state=42,
                                                   stratify=train_data_unsplit['income'])
    else:  # Standard split
        train_data_unsplit, test_data = sk_train_test_split(df, test_size=0.2, random_state=42, stratify=df['income'])
        train_data, val_data = sk_train_test_split(train_data_unsplit, test_size=val_size, random_state=42,
                                                   stratify=train_data_unsplit['income'])

    # Fit the preprocessor on the training data's features ONLY
    preprocessor.fit(train_data.drop('income', axis=1))

    # Apply the transformation
    X_train = preprocessor.transform(train_data.drop('income', axis=1))
    y_train = train_data['income'].values
    X_val = preprocessor.transform(val_data.drop('income', axis=1))
    y_val = val_data['income'].values
    X_test = preprocessor.transform(test_data.drop('income', axis=1))
    y_test = test_data['income'].values

    # Add noise to numeric features of the test set if specified
    if noise_intensity > 0:
        num_feature_count = len(numerical_features)
        noise = np.random.normal(0, noise_intensity, X_test[:, :num_feature_count].shape)
        X_test[:, :num_feature_count] += noise

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Create DataLoaders
    train_dataset = data_utils.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = data_utils.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = data_utils.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = data_utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = data_utils.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = data_utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    num_features = X_train_tensor.shape[1]
    print(f"Dataset processed. Number of features: {num_features}")

    return (train_loader, val_loader, test_loader), num_features