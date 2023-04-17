import torch as ch
from torch.optim import Adam
from inverseUtil import *


def get_Noise(model, args):
    if args.noise_type == 'gau':
        noise = gau_Noise(args)
    elif args.noise_type == 'phoni':
        noise = phoni_Noise(model, args)
    return noise


def add_Noise(rep, dummy_data, args):
    for k in range(len(rep)):
        rand_classes = np.random.choice(args.num_class, args.num_class, replace=False)
        rand_index = np.random.choice(args.num_class, args.phoni_num)
        dat = rep[k]
        for i in range(args.phoni_num):
            rand_dummy = rand_classes[i] + (rand_index[i] * args.num_class)
            dat = dat + (dummy_data[rand_dummy] * args.noise_scale)
        dat = dat / args.data_scale
        rep[k] = dat
    return rep


def phoni_Noise(model, args):
    dummy_data = ch.randn(args.noise_structure).to("cuda:0").requires_grad_(True).to(ch.float)
    dummy_label = ch.arange(0.0, args.num_class)
    dummy_label = dummy_label.repeat(1, args.phoni_size).reshape(-1).to("cuda:0").requires_grad_(True).to(ch.long)
    optimizer = Adam([dummy_data], 0.05)
    criterion = nn.CrossEntropyLoss()

    for iters in range(500):
        optimizer.zero_grad()
        dummy_pred = model.forward(dummy_data).to(ch.float)
        dummy_ce = - args.alpha * criterion(dummy_pred, dummy_label) \
                   + abs(abs(dummy_data).sum(axis=(1, 2, 3)).mean() - args.a.detach()) \
                   + abs(abs(dummy_data).sum(axis=(1, 2, 3)).var() - args.b.detach()) \
                   + abs(abs(dummy_data).var(axis=(1, 2, 3)).mean() - args.c.detach()) \
                   + abs(abs(dummy_data).var(axis=(1, 2, 3)).var() - args.d.detach())
        dummy_ce.backward()
        optimizer.step()
    # print(ch.softmax(model.forward(dummy_data).to(ch.float).cpu().detach(), dim=0)[0:5])
    return dummy_data


def gau_Noise(args):
    noise = torch.randn(args.noise_structure).to("cuda:0").requires_grad_(True).to(ch.float)
    return noise
