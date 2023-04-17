from Utils import helpers
from . import classifiers

import torch as ch
from torch import nn

from .encoder import *


def generate_Models(args):
    if args.dataset == 'cifar':
        encoder = Cifar_Encoder(num_step=args.attack_layer)
        classsifier = classifiers.Cifar10_Classifier(num_step=args.attack_layer)
    elif args.dataset == 'cifar100':
        encoder = Cifar_Encoder(num_step=args.attack_layer)
        classsifier = classifiers.Classifier(num_step=args.attack_layer)
    elif args.dataset == 'mnist':
        encoder = Mnist_Encoder(num_step=args.attack_layer)
        classsifier = classifiers.Mnist_Classifier(num_step=args.attack_layer)
    elif args.dataset == 'fashion_mnist':
        encoder = Mnist_Encoder(num_step=args.attack_layer)
        classsifier = classifiers.Mnist_Classifier(num_step=args.attack_layer)
    return encoder, classsifier


class enc_model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.encoder, self.classifier = generate_Models(args)
        if args.noise_knowledge != 'none':
            self.denoiser = Denoiser(args)
            self.noiser = Noiser(args)
        self.criterion = nn.CrossEntropyLoss().cuda()

    def forward(self, input, target):
        rep_out = self.encoder(input)
        #rep_out = self.noiser(rep_out)
        out = self.classifier(rep_out)
        loss = self.criterion(ch.sigmoid(out), target)
        acc = helpers.accuracy(out, target)[0]
        return out, loss, acc

    def rep_forward(self, rep_out, target):
        out = self.classifier(rep_out)
        loss = self.criterion(ch.sigmoid(out), target)
        acc = helpers.accuracy(out, target)[0]
        return loss, acc

    def get_rep(self, rep_out, target):
        out = self.encoder(rep_out)
        loss = self.criterion(ch.sigmoid(self.classifier(out)), target)
        acc = helpers.accuracy(self.classifier(out), target)[0]
        return out, loss, acc


class dec_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.decoder = Denoiser()
        self.criterion = nn.MSELoss().cuda()

    def forward(self, input, rep_out):
        est = self.decoder(rep_out)
        est = est.view(-1, 3, 32, 32)
        loss = self.criterion(ch.sigmoid(est), input)
        acc = helpers.accuracy(est, input)[0]
        return est, loss
