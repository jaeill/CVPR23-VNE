

import torch
import torchvision
import numpy as np
import time
import datetime
import os
from PIL import ImageOps, Image
import cv2
import argparse
import glob


parser = argparse.ArgumentParser(description='I_VNE training')

parser.add_argument('--cache_name', default='I_VNE_ImageNet_100', type=str, metavar='DIR', help='cache_name')
parser.add_argument('--cache_path', default='./cache', type=str, metavar='DIR', help='path to cache directory')
parser.add_argument('--gpu_num', default='0', type=str, metavar='N', help='gpu_num')
parser.add_argument('--datadir', default='./data/imagenet', type=str, metavar='DIR', help='path to data directory')
parser.add_argument('--subclass_file', default='./subclass_imgnet100.csv', type=str, metavar='FILE', help='subclass_file')
parser.add_argument('--num_workers', default=16, type=int, metavar='N', help='number of data loader workers')

parser.add_argument('--weight_decay', default=1.0e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--epochs', default=800, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=10, type=int, metavar='N', help='number of epochs to warmup')
parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--alpha_1', default=4.16, type=float, metavar='COEF', help='alpha_1') # 4.16 = log(64) = log(batch_size)
parser.add_argument('--alpha_2', default=1.00, type=float, metavar='COEF', help='alpha_2')
parser.add_argument('--base_learning_rate', default=0.40, type=float, metavar='LR', help='base learning rate')
parser.add_argument('--header_size', default=256, type=int, metavar='N', help='header_size')
parser.add_argument('--extra_views', default=4, type=int, metavar='N', help='extra_views') # total_views = 2 + extra_views

# Try not to stop and resume from checkpoint.
parser.add_argument('--from_checkpoint', dest='from_checkpoint', action='store_true')
##################################################




class Imgnet_subclass(torch.utils.data.Dataset):
    def __init__(self, root, subclass_file, transform=None):
        super(Imgnet_subclass, self).__init__()

        self.transform = transform

        subclass_list = []
        with open(subclass_file, 'r') as fid:
            for eachline in fid.read().splitlines():
                subclass_list += eachline.split(',')
        subclass_list.sort()

        self.imgs = []
        for idx, subclass in enumerate(subclass_list):
            file_list = glob.glob(os.path.join(root, subclass, '*.JPEG'))
            file_list.sort()
            self.imgs += [(filename, idx) for filename in file_list]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        filename, target_idx = self.imgs[index]
        with open(filename, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target_idx



class CosineAnnealingLR_Warmup(object):
    def __init__(self, optimizer, warmup_epochs, T_max, iter_per_epoch, base_lr, warmup_lr, eta_min, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.iter_per_epoch = iter_per_epoch
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.eta_min = eta_min
        self.last_epoch = last_epoch

        self.warmup_iter = self.iter_per_epoch * self.warmup_epochs
        self.cosine_iter = self.iter_per_epoch * (self.T_max - self.warmup_epochs)
        self.current_iter = (self.last_epoch + 1) * self.iter_per_epoch

        self.step()

    def get_current_lr(self):
        if self.current_iter < self.warmup_iter:
            current_lr = (self.base_lr - self.warmup_lr) / self.warmup_iter * self.current_iter + self.warmup_lr
        else:
            current_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + np.cos(np.pi * (self.current_iter-self.warmup_iter) / self.cosine_iter)) / 2
        return current_lr

    def step(self):
        current_lr = self.get_current_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr
        self.current_iter += 1



class MultiTransform(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list
    def __call__(self, input_img):
        return [each_transform(input_img) for each_transform in self.transform_list]


class Solarization(object):
    def __init__(self):
        pass

    def __call__(self, img):
        return ImageOps.solarize(img)


class GaussianBlur(object):
    def __init__(self, ksize, min_sigma=0.1, max_sigma=2.0):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        # kernel size is set to be 10% of the image height/width
        self.ksize = ksize

    def __call__(self, input_img):
        input_img = np.array(input_img)
        sigma = (self.max_sigma - self.min_sigma) * np.random.rand() + self.min_sigma
        input_img = Image.fromarray(cv2.GaussianBlur(input_img, (self.ksize, self.ksize), sigma))

        return input_img


def get_vne(H):
    Z = torch.nn.functional.normalize(H, dim=1)
    sing_val = torch.svd(Z / np.sqrt(Z.shape[0]))[1]
    eig_val = sing_val ** 2
    return - (eig_val * torch.log(eig_val)).nansum()


def train_I_VNE(args):
    print(args)

    # Use the same augmentation sets in SwAV
    imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

    transform_large1 = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.14, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        GaussianBlur(23),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*imagenet_norm)
    ])

    transform_large2 = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.14, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomApply([GaussianBlur(23)], p=0.1),
        torchvision.transforms.RandomApply([Solarization()], p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*imagenet_norm)
    ])

    transform_small1 = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(96, scale=(0.05, 0.14)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        GaussianBlur(9),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*imagenet_norm)
    ])

    transform_small2 = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(96, scale=(0.05, 0.14)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.RandomApply([GaussianBlur(9)], p=0.1),
        torchvision.transforms.RandomApply([Solarization()], p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*imagenet_norm)
    ])

    transform_list = [transform_large1,transform_large2] + [transform_small1,transform_small2] * int(args.extra_views/2.)
    args.extra_views = len(transform_list) - 2
    dataset_train = Imgnet_subclass(args.datadir + '/train', args.subclass_file, transform=MultiTransform(transform_list))
    loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)


    model = torchvision.models.resnet50(pretrained=False)
    rep_size = model.fc.in_features
    model.fc = torch.nn.Identity()

    projector = torch.nn.Sequential(torch.nn.Linear(in_features=rep_size, out_features=rep_size), torch.nn.BatchNorm1d(rep_size), torch.nn.ReLU(inplace=True), \
                                    torch.nn.Linear(in_features=rep_size, out_features=rep_size), torch.nn.BatchNorm1d(rep_size), torch.nn.ReLU(inplace=True), \
                                    torch.nn.Linear(in_features=rep_size, out_features=args.header_size), torch.nn.BatchNorm1d(args.header_size))


    last_epoch = 0
    cache_dict = dict()

    cache_file_name = '{0}/state_dict.{1}.pt'.format(args.cache_path, args.cache_name)
    if os.path.exists(cache_file_name) and args.from_checkpoint:
        cache_dict = torch.load(cache_file_name)
        last_epoch = cache_dict['epoch']
        model.load_state_dict(cache_dict['model'])
        projector.load_state_dict(cache_dict['projector'])
        print('epoch: {0} loaded'.format(last_epoch))

    model.to('cuda')
    projector.to('cuda')


    model = torch.nn.DataParallel(model)
    projector = torch.nn.DataParallel(projector)


    model.train()
    projector.train()

    params_with_wd = []
    params_without_wd = []

    for name, param in model.named_parameters():
        if 'bn' in name or 'bias' in name:
            params_without_wd.append(param)
        else:
            params_with_wd.append(param)

    for name, param in projector.named_parameters():
        if 'bn' in name or 'bias' in name:
            params_without_wd.append(param)
        else:
            params_with_wd.append(param)


    learning_rate = args.batch_size / 256. * args.base_learning_rate

    optimizer_with_wd = torch.optim.SGD(params_with_wd, lr = learning_rate, momentum = 0.9, weight_decay = args.weight_decay, nesterov = False)
    scheduler_with_wd = CosineAnnealingLR_Warmup(optimizer_with_wd, warmup_epochs=args.warmup_epochs, T_max=args.epochs, iter_per_epoch=len(loader_train), base_lr=learning_rate, warmup_lr=1e-6, eta_min=0, last_epoch=last_epoch-1)
    optimizer_with_wd.zero_grad();optimizer_with_wd.step()

    optimizer_without_wd = torch.optim.SGD(params_without_wd, lr = learning_rate, momentum = 0.9, weight_decay = 0, nesterov = False)
    scheduler_without_wd = CosineAnnealingLR_Warmup(optimizer_without_wd, warmup_epochs=args.warmup_epochs, T_max=args.epochs, iter_per_epoch=len(loader_train), base_lr=learning_rate, warmup_lr=1e-6, eta_min=0, last_epoch=last_epoch-1)
    optimizer_without_wd.zero_grad();optimizer_without_wd.step()


    if 'optimizer_with_wd' in cache_dict:
        optimizer_with_wd.load_state_dict(cache_dict['optimizer_with_wd'])
    if 'optimizer_without_wd' in cache_dict:
        optimizer_without_wd.load_state_dict(cache_dict['optimizer_without_wd'])

    tic = time.time()

    for epoch in range(last_epoch+1, args.epochs+1):
        print('epoch: {0} / lr: {1:.7f} / wd: {2:.7f} / {3}'.format(epoch, optimizer_with_wd.state_dict()['param_groups'][0]['lr'], args.weight_decay, env_info))
        loss_list = []
        cossim_list = []
        entropy_list = []
        for data, _ in loader_train:
            optimizer_with_wd.zero_grad()
            optimizer_without_wd.zero_grad()

            if args.extra_views > 0:
                proj_list = []
                proj_list.append(projector(model(torch.cat(data[:2]).to('cuda'))))
                proj_list.append(projector(model(torch.cat(data[2:]).to('cuda'))))
                projections = torch.cat(proj_list)
            else:
                projections = projector(model(torch.cat(data).to('cuda')))


            cossim_sum = 0.
            cossim_cnt = 0.
            for xii in range(2):
                for xjj in range(xii+1,args.extra_views+2):
                    cossim_sum += torch.nn.functional.cosine_similarity(projections[args.batch_size*xii:args.batch_size*(xii+1)], projections[args.batch_size*xjj:args.batch_size*(xjj+1)]).mean()
                    cossim_cnt += 1.
            avg_cossim = cossim_sum / cossim_cnt


            entropy_sum = 0.
            entropy_cnt = 0.
            for xii in range(args.extra_views+2):
                entropy_sum += get_vne(projections[args.batch_size*xii:args.batch_size*(xii+1)])
                entropy_cnt += 1.
            avg_entropy = entropy_sum / entropy_cnt

            loss = -(args.alpha_1 * avg_cossim + args.alpha_2 * avg_entropy)

            loss.backward()


            optimizer_with_wd.step()
            optimizer_without_wd.step()

            scheduler_with_wd.step()
            scheduler_without_wd.step()

            loss_list.append(loss.item())
            cossim_list.append(avg_cossim.item())
            entropy_list.append(avg_entropy.item())

        toc = time.time()
        print('Elapsed: {0:.1f}, Next: {1}, Finish: {2}'.format(toc-tic, (datetime.datetime.now() + datetime.timedelta(seconds=(toc-tic))).strftime("%Y%m%d %H:%M"),\
                                (datetime.datetime.now() + datetime.timedelta(seconds=(toc-tic) * (args.epochs - epoch))).strftime("%Y%m%d %H:%M")))
        print('Avg Loss: {0:.3f} / Avg Cossim: {1:.3f} / Avg Entropy: {2:.3f}'.format(np.mean(loss_list), np.mean(cossim_list), np.mean(entropy_list)))
        tic = toc

        torch.save({'epoch':epoch, 'model': model.module.state_dict(), 'projector': projector.module.state_dict(), 'optimizer_with_wd': optimizer_with_wd.state_dict(), 'optimizer_without_wd': optimizer_without_wd.state_dict(), 'args': args}, cache_file_name + '.tmp')
        os.rename(cache_file_name + '.tmp', cache_file_name)


    print('\nDone.')




if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs(args.cache_path, exist_ok=True)
    
    env_info = '{0}:{1}'.format(os.uname().nodename, args.gpu_num)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    torch.backends.cudnn.benchmark = True

    train_I_VNE(args)




