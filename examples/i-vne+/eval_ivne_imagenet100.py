

import torch
import torchvision
import time
import datetime
import os
import argparse
import glob
import numpy as np
from PIL import ImageOps, Image


parser = argparse.ArgumentParser(description='I_VNE linear evaluation')
parser.add_argument('--num_workers', default=8, type=int, metavar='N', help='number of data loader workers')
parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--warmup_epochs', default=0, type=int, metavar='N', help='number of epochs to warmup')
parser.add_argument('--batch_size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--learning_rate', default=0.3, type=float, metavar='LR', help='learning_rate')
parser.add_argument('--momentum', default=0.995, type=float, metavar='MMT', help='momentum')
parser.add_argument('--weight_decay', default=0.000001, type=float, metavar='W', help='weight decay')
parser.add_argument('--subclass_file', default='./subclass_imgnet100.csv', type=str, metavar='FILE', help='subclass_file')
parser.add_argument('--datadir', default='./data/imagenet', type=str, metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--cache_name', default='I_VNE_ImageNet_100', type=str, metavar='DIR', help='cache_name')
parser.add_argument('--cache_path', default='./cache/', type=str, metavar='DIR', help='path to cache directory')
parser.add_argument('--gpu_num', default='0', type=str, metavar='N', help='gpu_num')


##########


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



def lin_eval_I_VNE(args):

    print(args)

    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224, scale=(0.20, 1.0)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_train = Imgnet_subclass(args.datadir + '/train', args.subclass_file, transform_train)
    loader_train = torch.utils.data.DataLoader(dataset_train, shuffle=True, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

    transform_valid = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_valid = Imgnet_subclass(args.datadir + '/val', args.subclass_file, transform_valid)
    loader_valid = torch.utils.data.DataLoader(dataset_valid, shuffle=False, pin_memory=True, batch_size=args.batch_size, num_workers=args.num_workers)

    model = torchvision.models.resnet50(zero_init_residual=True, num_classes=100)


    cache_file_name = '{0}/state_dict.{1}.pt'.format(args.cache_path, args.cache_name)
    cache_dict = torch.load(cache_file_name)
    last_epoch = cache_dict['epoch']
    msg = model.load_state_dict(cache_dict['model'], strict=False)
    print(msg)
    print('epoch: {0} loaded'.format(last_epoch))

    model.to('cuda')


    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model.fc.weight.data.normal_(mean=0.0, std=0.010)
    model.fc.bias.data.zero_()


    optimizer_fclayer = torch.optim.SGD(parameters, lr = args.learning_rate, momentum = args.momentum, weight_decay = args.weight_decay, nesterov = False)
    scheduler_fclayer = CosineAnnealingLR_Warmup(optimizer_fclayer, warmup_epochs=args.warmup_epochs, T_max=args.epochs, iter_per_epoch=len(loader_train), base_lr=args.learning_rate, warmup_lr=1e-6, eta_min=0, last_epoch=-1)
    optimizer_fclayer.zero_grad();optimizer_fclayer.step()


    model.eval()

    best_acc1 = 0.
    best_acc5 = 0.

    import time
    tic = time.time()
    for epoch in range(1, args.epochs+1):
        print('epoch: {0} / lr_fclayer: {1:.6f} / {2}'.format(epoch,\
                        optimizer_fclayer.state_dict()['param_groups'][0]['lr'], args.env_info))

        for data, targets in loader_train:
            optimizer_fclayer.zero_grad()
            outputs = model(data.to('cuda'))

            loss = torch.nn.functional.cross_entropy(outputs, targets.to('cuda'))
            loss.backward()

            optimizer_fclayer.step()
            scheduler_fclayer.step()

        [correct_1, correct_5], total = valid(epoch, model, loader_valid)
        if best_acc1 < 1. * correct_1 / total:
            best_acc1 = 1. * correct_1 / total
        if best_acc5 < 1. * correct_5 / total:
            best_acc5 = 1. * correct_5 / total

        toc = time.time()
        print('best_acc1: {0:.3%} / best_acc5: {1:.3%}'.format(best_acc1, best_acc5))
        print('Elapsed: {0:.1f}, Next: {1}, Finish: {2}'.format(toc-tic,\
                                (datetime.datetime.now() + datetime.timedelta(seconds=(toc-tic))).strftime("%Y%m%d %H:%M"),\
                                (datetime.datetime.now() + datetime.timedelta(seconds=(toc-tic) * (args.epochs - epoch))).strftime("%Y%m%d %H:%M")))
        tic = toc


    print('\nDone.')



def valid(epoch, model, dataloader):
    if epoch >= 0:
        print('Validating Epoch: %d' % epoch)

    correct_1 = 0
    correct_5 = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            total += targets.size(0)

            predicted = outputs.topk(5,1,largest=True,sorted=True)[1]
            labels = targets.view(targets.size(0),-1).expand_as(predicted)
            correct = predicted.eq(labels).float()

            correct_5 += correct[:,:5].sum().item()
            correct_1 += correct[:,:1].sum().item()

    print('{0:.0f} / {1:.0f} / {2:.0f} / {3:.3%} / {4:.3%}'.format(correct_1, correct_5, total, 1. * correct_1 / total, 1. * correct_5 / total))

    return [correct_1, correct_5], total



if __name__ == '__main__':
    args = parser.parse_args()
    
    args.env_info = '{0}:{1}'.format(os.uname().nodename, args.gpu_num)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num
    torch.backends.cudnn.benchmark = True

    lin_eval_I_VNE(args)

