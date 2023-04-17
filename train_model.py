import copy

import numpy as np

from Attacks.attacks import attack, attack_inversion
from Defense.OurDefense import *
from torch.optim import Adam
from tqdm import tqdm as tqdm
from Utils.helpers import AverageMeter, get_mse
from skimage.metrics import structural_similarity as compare_ssim


def train_model(models, loaders, args):
    models.train()
    train_loader, val_loader = loaders
    opts = []
    enc_opt = Adam(models.encoder.parameters(), args.lr)
    def_opt = Adam(models.noiser.parameters(), args.lr)
    cla_opt = Adam(models.classifier.parameters(), args.lr)
    opts.append(enc_opt)
    opts.append(def_opt)
    opts.append(cla_opt)
    for i in range(0, args.epoch):
        train_loop(train_loader, models, opts, args, i, False)
    if args.attack_type == 'denoiser':
        for j in range(0, 5):
            opt = Adam(models.denoiser.parameters(), args.lr)
            denoise(train_loader, models, opt, args, j)
    for i in range(0, args.epoch):
        train_loop(train_loader, models, opts, args, i, True)
    # if args.attack_type == 'denoiser':
    #     for j in range(0, 5):
    #         opt = Adam(models.denoiser.parameters(), args.lr)
    #         denoise(train_loader, models, opt, args, j)
    eval_loop(val_loader, models, args)
    # train_decoder(train_loader, models, itr=500, lr=0.05)


def train_loop(train_loader, models, opts, args, epoch, defender):
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    enc_opt, def_opt, cla_opt = opts
    loss_enc = AverageMeter()
    acc_enc = AverageMeter()
    # a = []
    # b = []
    # c = []
    # d = []
    # if (epoch >= args.burn_in) and (args.noise_type != 'none'):
    #     dummy_data = get_Noise(models.classifier, args)
    for i, (input, target) in iterator:
        input = preprocess(input)
        rep_out = models.encoder.forward(input)
        # a.append(abs(rep_out).sum(axis=(1, 2, 3)).mean())
        # b.append(abs(rep_out).sum(axis=(1, 2, 3)).var())
        # c.append(abs(rep_out).var(axis=(1, 2, 3)).mean())
        # d.append(abs(rep_out).var(axis=(1, 2, 3)).var())
        # if (epoch >= args.burn_in) and (args.noise_type != 'none'):
        #     rep_out = add_Noise(rep_out, dummy_data, args)
        if defender:
            rep_out = models.noiser.forward(rep_out)
            loss, acc = models.rep_forward(rep_out, target)
            rep_gen = models.denoiser.forward(rep_out)
            decoder_loss = ((rep_gen - input) ** 2).mean()
            PSNR = 10 * torch.log10(1 ** 2 / decoder_loss)
            loss = 5 * loss - decoder_loss + PSNR
            def_opt.zero_grad()
            loss.backward()
            def_opt.step()
        else:
            loss, acc = models.rep_forward(rep_out, target)
            enc_opt.zero_grad()
            #def_opt.zero_grad()
            cla_opt.zero_grad()
            loss.backward()
            enc_opt.step()
            #def_opt.step()
            cla_opt.step()
        _, loss, acc = models.forward(input, target)
        loss_enc.update(loss.item(), input.size(0))
        acc_enc.update(acc.item(), input.size(0))
        desc = ('Epoch:{0} | '
                'Loss {Loss:.4f} | '
                'prec {prec:.4f} | '
        .format(
            epoch,
            Loss=loss_enc.avg,
            prec=acc_enc.avg))
        iterator.set_description(desc)
        iterator.refresh()
    # args.a = 10 * sum(a)/len(a)
    # args.b = sum(b)/len(b)
    # args.c = sum(c)/len(c)
    # args.d = sum(d)/len(d)
    #print(sum(a)/len(a), sum(c)/len(c), sum(b)/len(b), sum(d)/len(d))


def eval_loop(eval_loader, models, args):
    models.eval()
    iterator = tqdm(enumerate(eval_loader), total=len(eval_loader))
    loss_enc = AverageMeter()
    loss_dec = AverageMeter()
    acc_enc = AverageMeter()
    # if args.noise_type != 'none':
    #     dummy_data = get_Noise(models.classifier, args)
    # else:
    #     dummy_data = None
    target_images = []
    for i, (input, target) in iterator:
        input = preprocess(input)
        rep_out = models.encoder.forward(input)
        # if args.noise_type != 'none':
        #     rep_out = add_Noise(rep_out, dummy_data, args)
        rep_out = models.noiser.forward(rep_out)
        loss, acc = models.rep_forward(rep_out, target)
        ori_rep = models.encoder.forward(input)
        rep_gen = models.denoiser.forward(rep_out)
        decoder_loss = ((input - ori_rep) ** 2).mean()
        loss_enc.update(loss.item(), input.size(0))
        loss_dec.update(decoder_loss.item(), input.size(0))
        acc_enc.update(acc.item(), input.size(0))
        desc = ('Eval    | '
                'Loss {Loss:.4f} | '
                'denoiser loss {dec:.4f} | '
                'prec {prec:.4f} | '
        .format(
            Loss=loss_enc.avg,
            dec=loss_dec.avg,
            prec=acc_enc.avg))
        iterator.set_description(desc)
        iterator.refresh()
        target_images.append(ch.unsqueeze(input[0], 0))
        target_images.append(ch.unsqueeze(input[1], 0))
    if args.attack_type != 'none':
        atk_images, MSEs, SSIMs, PSNRs = [], [], [], []
        dummy_data = None
        for j in range(args.num_attacked):
            if args.attack_type == 'inversion' and args.noise_knowledge == 'none':
                atk_image, MSE, SSIM, PSNR = attack(target_images[j], models, args, dummy_data, j)
            else:
                atk_image, MSE, SSIM, PSNR = attack(target_images[0], models, args, dummy_data, j)
            atk_images.append(atk_image)
            MSEs.append(MSE)
            SSIMs.append(SSIM)
            PSNRs.append(PSNR)
        print('The mean MSE is : ', np.round(np.mean(MSEs), 2))
        print('The mean SSIM is : ', np.round(np.mean(SSIMs), 2))
        print('The mean PSNR is : ', np.round(np.mean(PSNRs), 2))


def denoise(train_loader, models, opt, args, epoch):
    models.eval()
    models.denoiser.train()
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    loss_enc = AverageMeter()
    #dummy_data = get_Noise(models.classifier, args)
    for i, (input, target) in iterator:
        rep_out = models.encoder.forward(input)
        #rep_out = add_Noise(rep_out, dummy_data, args)
        rep_out = models.noiser.forward(rep_out)
        rep_gen = models.denoiser.forward(rep_out)
        ori_rep = models.encoder.forward(input)
        decoder_loss = ((rep_gen - input) ** 2).mean()
        opt.zero_grad()
        decoder_loss.backward()
        opt.step()
        loss_enc.update(decoder_loss.item(), input.size(0))
        desc = ('Denoiser:{0} | '
                'Loss {Loss:.4f} | '
        .format(
            epoch,
            Loss=loss_enc.avg))
        iterator.set_description(desc)
        iterator.refresh()