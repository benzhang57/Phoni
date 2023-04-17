import os
from argparse import ArgumentParser

import cox
import numpy.random

from Utils.datasets import DATASETS
from Utils import helpers
import time
from Utils.models import enc_model, dec_model
from train_model import train_model, eval_loop

parser = ArgumentParser()
parser.add_argument('--dataset', choices=['cifar', 'mnist', 'fashion_mnist', 'cifar100'], default='cifar')
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--burn_in', type=int, default=5)
parser.add_argument('--attack_layer', type=int, default=0)
parser.add_argument('--attack_type', choices=['none', 'inversion', 'denoiser', 'Bayes'], default='none')
parser.add_argument('--attacker_knowledge', choices=['black-box', 'white-box', 'input'], default='white-box')
parser.add_argument('--noise_knowledge', choices=['none', 'exist', 'pattern', 'exact'], default='none')
parser.add_argument('--noise_type', choices=['none', 'phoni', 'gau'], default='none')
parser.add_argument('--alpha', type=float, default=1.5)
parser.add_argument('--beta', type=float, default=0.005)
parser.add_argument('--lam', type=float, default=0.01)
parser.add_argument('--noise_scale', type=float, default=10.0)
parser.add_argument('--data_scale', type=float, default=1.25)
parser.add_argument('--data_aug', type=bool, default=False)
parser.add_argument('--save_images', type=bool, default=False)
parser.add_argument('--phoni_num', type=int, default=3)
parser.add_argument('--phoni_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--noise_structure', default=[1000, 32, 32, 32])
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--num_attacked', type=int, default=10)
parser.add_argument('--attack_epoch', type=int, default=1000)
parser.add_argument('--image_names', default='default')
parser.add_argument('--a', type=float, default=0)
parser.add_argument('--b', type=float, default=0)
parser.add_argument('--c', type=float, default=0)
parser.add_argument('--d', type=float, default=0)

args = parser.parse_args()


def arg_helper(args):
    if args.dataset == "cifar":
        args.num_class = 10
        if args.attack_layer == 0 or args.attack_layer == 1:
            args.noise_structure = [1000, 32, 32, 32]
        elif args.attack_layer == 2 or args.attack_layer == 5:
            args.noise_structure = [1000, 64, 32, 32]
        elif args.attack_layer == 3 or args.attack_layer == 4:
            args.noise_structure = [1000, 128, 32, 32]
        elif args.attack_layer == 6:
            args.noise_structure = [1000, 65536]
        elif args.attack_layer == 7:
            args.noise_structure = [1000, 1024]
    if args.dataset == "mnist" or args.dataset == "fashion_mnist":
        args.num_class = 10
        if args.attack_layer == 0 or args.attack_layer == 1:
            args.noise_structure = [1000, 8, 28, 28]
        elif args.attack_layer == 2:
            args.noise_structure = [1000, 3, 28, 28]
        elif args.attack_layer == 3:
            args.noise_structure = [1000, 4*28*28]
        elif args.attack_layer == 4:
            args.noise_structure = [1000, 1024]
        elif args.attack_layer == 5:
            args.noise_structure = [1000, 64]
    return args


def main(args):
    print(args)
    data_path = os.path.expandvars(args.dataset)
    dataset = DATASETS[args.dataset](data_path)
    numpy.random.RandomState(42)
    train_loader, val_loader = dataset.make_loaders(8, args.batch_size, data_aug=args.data_aug)
    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)
    enc_Model = enc_model(args).to(args.device)
    starting_time = time.time()
    train_model(enc_Model, loaders, args)
    end_time = time.time()
    total_time = end_time - starting_time
    print('Total Time: ', total_time)


if __name__ == '__main__':
    args = cox.utils.Parameters(args.__dict__)
    args = arg_helper(args)
    main(args)
