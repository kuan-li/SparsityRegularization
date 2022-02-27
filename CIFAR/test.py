
import sys
import os
import pickle
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn

from PIL import Image as PILImage
from models.wrn_prime import WideResNet
from models.allconv import AllConvNet
from skimage.filters import gaussian as gblur

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std
    import utils.svhn_loader as svhn
    import utils.lsun_loader as lsun_loader
    import utils.score_calculation as lib

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_wrn_baseline', help='Method name.')
parser.add_argument('--arch', '-a', type=str, default='wrn', choices=['allconv', 'wrn'], help='Choose architecture.')
parser.add_argument('--machine', type=str, default='local', choices=['acm', 'local'], help='Choose machine.')


# Loading details
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--load', '-l', type=str, default='./snapshots/', help='Checkpoint path to resume / test.')
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save score.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

# EG and benchmark details
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
parser.add_argument('--score', type=str, default='energy', choices=['MSP', 'energy', 'M', 'xent', 'Odin'], help='score options.')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
parser.add_argument('--noise', type=float, default=0, help='noise for Odin')
args = parser.parse_args()

print(args)
# torch.manual_seed(1)
# np.random.seed(1)

# mean and standard deviation of channels of CIFAR-10 images
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

if args.machine == 'acm':
    data_path = '/opt/data/private/ood/data/'
    cifar_path = data_path + 'cifar'
    dtd_path = data_path + 'dtd/images'
    svhn_path =data_path + 'svhn/'
    places365_path = data_path + 'places365'
    lsun_c_path = data_path + 'LSUN_C'
    lsun_r_path = data_path + 'LSUN_resize'
    isun_path = data_path + 'iSUN'
    tiny_imagenet_path = data_path + 'tiny-imagenet-200/test/'

if args.machine == 'local':
    data_path = '/data1/church/ood/data/'
    cifar_path = data_path + 'cifar'
    dtd_path = data_path + 'dtd/images'
    svhn_path =data_path + 'svhn/'
    places365_path = data_path + 'places365'
    lsun_c_path = data_path + 'LSUN_C'
    lsun_r_path = data_path + 'LSUN_resize'
    isun_path = data_path + 'iSUN'
    tiny_imagenet_path = data_path + 'tiny-imagenet-200/test/'


if 'cifar10_' in args.method_name:
    test_data = dset.CIFAR10(cifar_path, train=False, transform=test_transform)
    num_classes = 10
else:
    test_data = dset.CIFAR100(cifar_path, train=False, transform=test_transform)
    num_classes = 100


test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)


# Create model
if args.arch == 'allconv':
    net = AllConvNet(num_classes)
# elif args.arch == 'resnet':
#     net = ResNet34(num_c=num_classes)
# elif args.arch == 'densenet':
#     net = DenseNet3(100, int(num_classes))
elif args.arch == 'wrn':
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
else:
    raise Exception('unknown network architecture: {}'.format(args.model))

    
print('\nThe number of model parameters: {}\n'.format(sum([p.data.nelement() for p in net.parameters()])))

start_epoch = 0

# Restore model
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        if 'pretrained' in args.method_name:
            subdir = 'pretrained'

        elif 'oe_tune' in args.method_name:
            subdir = 'oe_tune'

        elif 'tune' in args.method_name:
            # subdir = 'tune_sroe'
            subdir = 'tune_sr'
            # subdir = 'tune_test1'
            # subdir = 'tune_test2'

        elif 'energy_ft' in args.method_name:
            subdir = 'energy_ft'

        elif 'oe_scratch' in args.method_name:
            subdir = 'oe_scratch'

        elif 'baseline' in args.method_name:
            subdir = 'baseline'

        else:
            raise Exception('unknown subdir: {}'.format(args.subdir))

        model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '_epoch_' + str(i) + '.pt')
        # print(model_name)
        # exit(0)
        # model_name = os.path.join(os.path.join(args.load, subdir), args.method_name + '_best' + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume "+model_name

net.eval()

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    net.cuda()
    # torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

# /////////////// Detection Prelims ///////////////
ood_num_examples = len(test_data) // 5
expected_ap = ood_num_examples / (ood_num_examples + len(test_data))

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_ood_scores(loader, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // args.test_bs and in_dist is False:
                break

            data = data.cuda()
            output, vector_feature = net(data)
            smax = to_np(F.softmax(output, dim=1))

            # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
            if args.score == 'MSP':
                _score.append(-np.max(smax, axis=1))
                # _right_score.append(-np.max(smax[right_indices], axis=1))
                # _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

            if args.score == 'energy':
                _score.append(-to_np((args.T*torch.logsumexp(output / args.T, dim=1))))
            
            elif args.score == 'xent': 
                _score.append( to_np( -1 * (F.softmax(output, dim=1) * F.log_softmax(output, dim=1)).sum(1)) )

            else: # original MSP and Mahalanobis (but Mahalanobis won't need this returned)
                _score.append(-np.max(smax, axis=1))

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)
                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
       
    if in_dist:    
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    
    else:
        return concat(_score)[:ood_num_examples].copy()


if args.score == 'Odin':
    # separated because no grad is not applied
    in_score, right_score, wrong_score = lib.get_ood_scores_odin(test_loader, net, args.test_bs, ood_num_examples, args.T, args.noise, in_dist=True)


elif args.score == 'M':
    from torch.autograd import Variable
    _, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

    if 'cifar10_' in args.method_name:
        train_data = dset.CIFAR10(cifar_path, train=True, transform=test_transform)
    else:
        train_data = dset.CIFAR100(cifar_path, train=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
    num_batches = ood_num_examples // args.test_bs

    temp_x = torch.rand(2,3,32,32)
    temp_x = Variable(temp_x)
    temp_x = temp_x.cuda()
    temp_list = net.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0

    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance', count)
    # print(feature_list) # 特征的维数
    # exit(0)
    sample_mean, precision = lib.sample_estimator(net, num_classes, feature_list, train_loader)
    in_score, _ = lib.get_Mahalanobis_score(net, test_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches, in_dist=True)


else:
    in_score, right_score, wrong_score = get_ood_scores(test_loader, in_dist=True)

# print(in_score[:10])
# exit(0)


num_right = len(right_score)
num_wrong = len(wrong_score)

print('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

# /////////////// End Detection Prelims ///////////////
print('\nUsing CIFAR-10 as typical data') if num_classes == 10 else print('\nUsing CIFAR-100 as typical data')


# /////////////// Error Detection ///////////////
print('\n\nOut of Distribution Detection: ')

# /////////////// OOD Detection ///////////////
auroc_list, aupr_list, fpr_list = [], [], []

def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg):

    aurocs, auprs, fprs = [], [], []

    for _ in range(num_to_avg):
        if args.score == 'Odin':
            out_score = lib.get_ood_scores_odin(ood_loader, net, args.test_bs, ood_num_examples, args.T, args.noise)
        elif args.score == 'M':
            out_score, _ = lib.get_Mahalanobis_score(net, ood_loader, num_classes, sample_mean, precision, count-1, args.noise, num_batches)

        else:
            out_score = get_ood_scores(ood_loader)
        
        if args.out_as_pos: # OE's defines out samples as positive
            measures = get_measures(out_score, in_score)
        else:
            measures = get_measures(-in_score, -out_score)
        aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])
    
    print(in_score[:3], out_score[:3])
    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
    auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

    if num_to_avg >= 5:
        print_measures_with_std(aurocs, auprs, fprs, args.method_name)
    else:
        print_measures(auroc, aupr, fpr, args.method_name)

# /////////////// Textures ///////////////
ood_data = dset.ImageFolder(root=dtd_path,
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=4, pin_memory=True)
print('\n\nTexture Detection')
get_and_print_results(ood_loader)


# /////////////// SVHN /////////////// # cropped and no sampling of the test set
ood_data = svhn.SVHN(root=svhn_path, split="test",
                     transform=trn.Compose(
                         [#trn.Resize(32),
                         trn.ToTensor(), trn.Normalize(mean, std)]), download=False)
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nSVHN Detection')
get_and_print_results(ood_loader)

# /////////////// Places365 ///////////////
ood_data = dset.ImageFolder(root=places365_path,
                            transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                   trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=2, pin_memory=True)
print('\n\nPlaces365 Detection')
get_and_print_results(ood_loader)

# /////////////// LSUN-C ///////////////
ood_data = dset.ImageFolder(root=lsun_c_path,
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\nLSUN_C Detection')
get_and_print_results(ood_loader)

# /////////////// LSUN-R ///////////////
ood_data = dset.ImageFolder(root=lsun_r_path,
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\nLSUN_Resize Detection')
get_and_print_results(ood_loader)

# /////////////// iSUN ///////////////
ood_data = dset.ImageFolder(root=isun_path,
                            transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                         num_workers=1, pin_memory=True)
print('\n\niSUN Detection')
get_and_print_results(ood_loader)


# /////////////// Mean Results ///////////////

print('\n\nMean Test Results!!!!!')
print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name)