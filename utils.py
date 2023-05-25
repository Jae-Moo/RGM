# ---------------------------------------------------------------
# This file has been modified from following sources: 
# Source:
# 1. https://github.com/pytorch/vision/blob/ea6b879e90459006e71a164dc76b7e2cc3bff9d9/torchvision/datasets/lsun.py (BSD 3-Clause License)
# 2. https://github.com/NVlabs/LSGM/blob/main/util/ema.py (NVIDIA License)
# 3. https://github.com/NVlabs/denoising-diffusion-gan/blob/main/train_ddgan.py (NVIDIA License)
# ---------------------------------------------------------------

import os
import warnings
import numpy as np
import torch
from torch.optim import Optimizer
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from PIL import Image
import io
from torchvision.datasets.vision import VisionDataset
import os.path
from collections.abc import Iterable
import pickle
from torchvision.datasets.utils import verify_str_arg, iterable_to_str

# ------------------------
# EMA
# ------------------------
class EMA(Optimizer):
    def __init__(self, opt, ema_decay):
        '''
        EMA Codes adapted from https://github.com/NVlabs/LSGM/blob/main/util/ema.py
        '''
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.
        self.optimizer = opt
        self.state = opt.state
        self.param_groups = opt.param_groups

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        # stop here if we are not applying EMA
        if not self.apply_ema:
            return retval

        ema, params = {}, {}
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]

                # State initialization
                if 'ema' not in state:
                    state['ema'] = p.data.clone()

                if p.shape not in params:
                    params[p.shape] = {'idx': 0, 'data': []}
                    ema[p.shape] = []

                params[p.shape]['data'].append(p.data)
                ema[p.shape].append(state['ema'])

            for i in params:
                params[i]['data'] = torch.stack(params[i]['data'], dim=0)
                ema[i] = torch.stack(ema[i], dim=0)
                ema[i].mul_(self.ema_decay).add_(params[i]['data'], alpha=1. - self.ema_decay)

            for p in group['params']:
                if p.grad is None:
                    continue
                idx = params[p.shape]['idx']
                self.optimizer.state[p]['ema'] = ema[p.shape][idx, :]
                params[p.shape]['idx'] += 1

        return retval

    def load_state_dict(self, state_dict):
        super(EMA, self).load_state_dict(state_dict)
        # load_state_dict loads the data to self.state and self.param_groups. We need to pass this data to
        # the underlying optimizer too.
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def swap_parameters_with_ema(self, store_params_in_ema):
        """ This function swaps parameters with their ema values. It records original parameters in the ema
        parameters, if store_params_in_ema is true."""

        # stop here if we are not applying EMA
        if not self.apply_ema:
            warnings.warn('swap_parameters_with_ema was called when there is no EMA weights.')
            return

        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if not p.requires_grad:
                    continue
                ema = self.optimizer.state[p]['ema']
                if store_params_in_ema:
                    tmp = p.data.detach()
                    p.data = ema.detach()
                    self.optimizer.state[p]['ema'] = tmp
                else:
                    p.data = ema.detach()


# ------------------------
# Dataset
# ------------------------

class CelebA_HQ(data.Dataset):
    '''Note: CelebA (about 200000 images) vs CelebA-HQ (30000 images)'''
    def __init__(self, root, partition_path, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform

        # Split train/val/test 
        self.partition_dict = {}
        self.get_partition_label(partition_path)
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.save_img_path()
        print('[Celeba-HQ Dataset]')
        print(f'Train {len(self.train_dataset)} | Val {len(self.val_dataset)} | Test {len(self.test_dataset)}')

        if mode == 'train':
            self.dataset = self.train_dataset
        elif mode == 'val':
            self.dataset = self.val_dataset
        elif mode == 'test':
            self.dataset = self.test_dataset
        else:
            raise ValueError

    def get_partition_label(self, list_eval_partition_celeba_path):
        '''Get partition labels (Train 0, Valid 1, Test 2) from CelebA
        See "celeba/Eval/list_eval_partition.txt"
        '''
        with open(list_eval_partition_celeba_path, 'r') as f:
            for line in f.readlines():
                filenum = line.split(' ')[0].split('.')[0] # Use 6-digit 'str' instead of int type
                partition_label = int(line.split(' ')[1]) # 0 (train), 1 (val), 2 (test)
                self.partition_dict[filenum] = partition_label

    def save_img_path(self):
        for filename in os.listdir(self.root):
            assert os.path.isfile(os.path.join(self.root, filename))
            filenum = filename.split('.')[0]
            label = self.partition_dict[filenum]
            if label == 0:
                self.train_dataset.append(os.path.join(self.root, filename))
            elif label == 1:
                self.val_dataset.append(os.path.join(self.root, filename))
            elif label == 2:
                self.test_dataset.append(os.path.join(self.root, filename))
            else:
                raise ValueError

    def __getitem__(self, index):
        img_path = self.dataset[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.dataset)


class LSUNClass(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        import lmdb
        super(LSUNClass, self).__init__(root, transform=transform,
                                        target_transform=target_transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        # cache_file = '_cache_' + ''.join(c for c in root if c in string.ascii_letters)
        # av begin
        # We only modified the location of cache_file.
        cache_file = os.path.join(self.root, '_cache_')
        # av end
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key, _ in txn.cursor()]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        img, target = None, -1
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length


class LSUN(VisionDataset):
    def __init__(self, root, classes='train', transform=None, target_transform=None):
        super(LSUN, self).__init__(root, transform=transform,
                                   target_transform=target_transform)
        self.classes = self._verify_classes(classes)

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c in self.classes:
            self.dbs.append(LSUNClass(
                root=root + '/' + c + '_lmdb',
                transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count

    def _verify_classes(self, classes):
        categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                      'conference_room', 'dining_room', 'kitchen',
                      'living_room', 'restaurant', 'tower', 'cat']
        dset_opts = ['train', 'val', 'test']

        try:
            verify_str_arg(classes, "classes", dset_opts)
            if classes == 'test':
                classes = [classes]
            else:
                classes = [c + '_' + classes for c in categories]
        except ValueError:
            if not isinstance(classes, Iterable):
                msg = ("Expected type str or Iterable for argument classes, "
                       "but got type {}.")
                raise ValueError(msg.format(type(classes)))

            classes = list(classes)
            msg_fmtstr = ("Expected type str for elements in argument classes, "
                          "but got type {}.")
            for c in classes:
                verify_str_arg(c, custom_msg=msg_fmtstr.format(type(c)))
                c_short = c.split('_')
                category, dset_opt = '_'.join(c_short[:-1]), c_short[-1]

                msg_fmtstr = "Unknown value '{}' for {}. Valid values are {{{}}}."
                msg = msg_fmtstr.format(category, "LSUN class",
                                        iterable_to_str(categories))
                verify_str_arg(category, valid_values=categories, custom_msg=msg)

                msg = msg_fmtstr.format(dset_opt, "postfix", iterable_to_str(dset_opts))
                verify_str_arg(dset_opt, valid_values=dset_opts, custom_msg=msg)

        return classes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self):
        return self.length

    def extra_repr(self):
        return "Classes: {classes}".format(**self.__dict__)


# Toy
def gmm(N):
    n = N//8
    centers = [[2*np.cos(i*np.pi/4), 2*np.sin(i*np.pi/4)] for i in range(8)]
    cov = [[1/200,0], [0,1/200]]
    dataset = np.hstack([[np.random.multivariate_normal(mean, cov, n)] for mean in centers])[0]
    return dataset

def gaussian25(N):
    n = N//25
    centers = [[i-2,j-2] for i in range(5) for j in range(5)]
    cov = [[1/4000,0], [0,1/4000]]
    dataset = np.hstack([[np.random.multivariate_normal(mean, cov, n)] for mean in centers])[0]
    return dataset

def checkerboard(N):
    n_points = 2*N
    n_classes = 2
    n = 8 # 4x4 checkerboard

    x = np.random.uniform(-2, 2, size=(n_points, n_classes))
    mask = np.logical_or(np.logical_and(np.sin(np.pi*x[:,0]) > 0.0, np.sin(np.pi*x[:,1]) > 0.0), \
    np.logical_and(np.sin(np.pi*x[:,0]) < 0.0, np.sin(np.pi*x[:,1]) < 0.0))
    idx = np.where(mask==False)
    dataset = np.delete(x,idx,0)
    return dataset


class Toydataset(Dataset):
    def __init__(self, name, N):
        self.N = N
        if name.lower()=='gmm':
            self.dataset = gmm(N)   
        elif name.lower()=='gaussian25':
            self.dataset = gaussian25(N)
        elif name.lower() == 'checkerboard':
            self.dataset = checkerboard(N)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


# get dataloader
def get_dataloader(args):
    num_workers = 4
    if args.dataset == 'mnist':
        dataset = MNIST('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5), (0.5))]), download=True)
    
    elif args.dataset == 'cifar10':
        dataset = CIFAR10('./data', train=True, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
    
    elif args.dataset == 'lsun':
        train_transform = transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.CenterCrop(args.image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ])
        train_data = LSUN(root='data', classes=['church_outdoor_train'], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)
    
    elif args.dataset == 'celeba_256':
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = CelebA_HQ(
            root='data/celeba-hq/celeba-256',
            partition_path='data/celeba-hq/list_eval_partition_celeba.txt',
            mode='train', # 'train', 'val', 'test'
            transform=train_transform,
        )
    else:
        num_workers = 0
        dataset = Toydataset(args.dataset, 4000)
        
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    return data_loader


def get_testloader(args):
    num_workers = 4
    if args.dataset == 'cifar10':
        train_data = CIFAR10('./data', train=False, transform=transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]), download=True)
        subset = list(range(0, 300))
        dataset = torch.utils.data.Subset(train_data, subset)
    elif args.dataset == 'lsun':
        train_transform = transforms.Compose([
                        transforms.Resize(args.image_size),
                        transforms.CenterCrop(args.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                    ])
        train_data = LSUN(root='data', classes=['church_outdoor_val'], transform=train_transform)
        subset = list(range(0, 300))
        dataset = torch.utils.data.Subset(train_data, subset)
    elif args.dataset == 'celeba_256':
        train_transform = transforms.Compose([
                transforms.Resize(args.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
        dataset = CelebA_HQ(
            root='data/celeba-hq/celeba-256',
            partition_path='data/celeba-hq/list_eval_partition_celeba.txt',
            mode='val', # 'train', 'val', 'test'
            transform=train_transform,
        )
        subset = list(range(0, 300))
        dataset = torch.utils.data.Subset(dataset, subset)
        
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )
    return data_loader


# ------------------------
# Get Model
# ------------------------

# Get pretrained model
def get_model(args, ckpt_path):
    from models.ncsnpp_generator_adagn import NCSNpp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    netG = NCSNpp(args).to(device)
    netG = torch.nn.DataParallel(netG, device_ids=[0,1,2,3])
    checkpoint = torch.load(ckpt_path, map_location=device)
    netG.load_state_dict(checkpoint)
    for p in netG.parameters():
        p.requires_grad = False
    return netG


# ------------------------
# Get loss
# ------------------------

# MMD distance

def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)
    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()
    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)
    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape
    
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
    
    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e
    
    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))
    return mmd2

def MMD(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


# Distributed Sliced Wasserstein Distance

def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))

def DSWD(first_samples, second_samples, num_projections, f, f_op, p=2, max_iter=10, lam=10, device="cuda"):
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    
    for _ in range(max_iter):
        projections = f(pro)
        cos = cosine_distance_torch(projections, projections)
        reg = lam * cos
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
        wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
                                        - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
        wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()
        
    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    
    
    wasserstein_distance = torch.abs((torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
                                    - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]))
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1), 1.0 / p)
    wasserstein_distance = torch.pow(torch.pow(wasserstein_distance, p).mean(), 1.0 / p)
    
    return wasserstein_distance
