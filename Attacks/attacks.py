from tqdm import tqdm as tqdm
from matplotlib import pyplot
from Utils.models import dec_model
from Utils.helpers import AverageMeter, get_PSNR, get_mse
from Defense.OurDefense import *
import torch as ch
from torch.optim import Adam, LBFGS
from skimage.metrics import structural_similarity as compare_ssim


def attack(input, models, args, noise, image_num):
    rep = get_attack_rep(input, models, args, noise)
    # if args.attack_type == "inversion":
    #     xGen = attack_inversion(input, rep, models.encoder, args, image_num)
    # if args.attack_tpye == "denoiser":
    #     xGen = models.denoiser.forward(rep)
    repGen = models.denoiser.forward(rep)
    #xGen = attack_inversion(input, repGen, models.encoder, args, image_num)
    MSE, SSIM, PSNR = draw_output(input, repGen, image_num, args)
    return repGen, MSE, SSIM, PSNR


def get_attack_rep(input, models, args, noise):
    rep = models.encoder.forward(input)
    rep = models.noiser.forward(rep)
    #rep = add_Noise(rep, noise, args).detach()
    # if args.noise_knowledge == 'exist':
    #     attack_noise = gau_Noise(args)
    #     rep = attack_denoise(rep, attack_noise, args)
    # elif args.noise_knowledge == 'pattern':
    #     attack_noise = get_Noise(models.classifier, args)
    #     rep = attack_denoise(rep, attack_noise, args)
    # elif args.noise_knowledge == 'exact':
    #     attack_noise = noise
    #     rep = attack_denoise(rep, attack_noise, args)
    return rep


def attack_denoise(rep, noise, args):
    for k in range(len(rep)):
        rand_classes = np.random.choice(args.num_class, args.num_class, replace=False)
        rand_index = np.random.choice(args.phoni_num, args.num_class)
        rand_dummy = rand_classes + (rand_index * args.num_class)
        dat = rep[k]
        dat = dat * args.data_scale
        for i in range(args.phoni_num):
            dat = dat - (noise[rand_dummy[i]]*args.noise_scale)
        rep[k] = dat
    return rep


def draw_output(input, xGen, image_num, args):
    input = deprocess(input)
    xGen = deprocess(xGen)
    if args.dataset == 'cifar' or args.dataset == 'cifar100':
        permu = (1, 2, 0)
    else:
        permu = (0, 1)
    ori_image = ch.squeeze(input * 255).cpu().detach().permute(permu).numpy().astype('uint8')
    atk_image = ch.squeeze(xGen.detach() * 255).cpu().detach().permute(permu).numpy().astype('uint8')
    pyplot.imshow(ori_image)
    ori_path = './Output/' + args.dataset + '/' + args.image_names + str(image_num) + '_ori.png'
    pyplot.savefig(ori_path)
    pyplot.imshow(atk_image)
    atk_path = './Output/' + args.dataset + '/' + args.image_names + str(image_num) + '_atk.png'
    pyplot.savefig(atk_path)
    MSE = np.round(get_mse(ori_image, atk_image), 2)
    if args.dataset == 'cifar':
        SSIM = np.round(compare_ssim(ori_image, atk_image, channel_axis=2), 2)
    else:
        SSIM = np.round(compare_ssim(ori_image, atk_image), 2)
    PSNR = np.round(get_PSNR(input, xGen), 2)
    #print("The MSE for the attack is: ", MSE)
    #print("The SSIM for the attack is: ", SSIM)
    #print("The PSNR for the attack is: ", PSNR)
    return MSE, SSIM, PSNR


def attack_inversion(input, rep_out, model, args, image_num, lambda_TV=0.5, lambda_l2=0.5):
    xGen = ch.zeros(input.size(), requires_grad=True, device="cuda")
    optimizer = Adam(params=[xGen], lr=0.05, eps=1e-3, amsgrad=True)
    ch.autograd.set_detect_anomaly(True)
    for i in range(args.attack_epoch+1):
        optimizer.zero_grad()
        xFeature = model.forward(xGen)
        featureLoss = ((xFeature - rep_out) ** 2).mean()
        TVLoss = TV(xGen)
        normLoss = l2loss(xGen)
        totalLoss = featureLoss + lambda_TV * TVLoss + lambda_l2 * normLoss  # - 1.0 * conv1Loss
        totalLoss.backward(retain_graph=True)
        optimizer.step()
        if (i % 1000 == 0):
            print("Iter ", i, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ",
                  TVLoss.cpu().detach().numpy(),
                  "l2Loss: ", normLoss.cpu().detach().numpy())
    return xGen


def attack_Rep(input, encoder):
    dummy_data = ch.randn(input.size()).to("cuda:0").requires_grad_(True).to(ch.float)
    dummy_label = encoder.forward(input.detach())
    optimizer = Adam([dummy_data], 0.1)
    criterion = nn.CrossEntropyLoss()

    for iters in range(5000):
        optimizer.zero_grad()
        dummy_pred = encoder.forward(dummy_data.detach()).to(ch.float)
        dummy_ce = criterion(dummy_pred, dummy_label.detach())
        dummy_ce.backward()
        optimizer.step()
    pyplot.imshow(ch.squeeze(input * 255).cpu().permute(1, 2, 0).numpy().astype('uint8'))
    pyplot.savefig('ori.png')
    pyplot.imshow(ch.squeeze(dummy_data.detach() * 255).cpu().permute(1, 2, 0).numpy().astype('uint8'))
    pyplot.savefig('atk.png')
    return dummy_data


def attack_Dlg(input, model, target):
    criterion = cross_entropy_for_onehot
    input = input.requires_grad_(True)
    pred, _, _ = model.forward(input, target)
    gt_onehot = label_to_onehot(target)
    y = criterion(pred, gt_onehot)
    dy_dx = ch.autograd.grad(y, model.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    dummy_data = ch.randn(input.size()).to("cuda:0").requires_grad_(True).to(ch.float)
    # dummy_label = ch.randn(gt_onehot.size()).to("cuda:0").requires_grad_(True).to(ch.float)
    dummy_label = gt_onehot.to("cuda:0").requires_grad_(True).to(ch.float)
    optimizer = LBFGS([dummy_data, dummy_label])
    for iters in range(500):
        def closure():
            optimizer.zero_grad()

            pred, _, _ = model.forward(dummy_data, target)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(pred,
                                   dummy_onehot_label)  # TODO: fix the gt_label to dummy_label in both code and slides.
            dummy_dy_dx = ch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):  # TODO: fix the variablas here
                grad_diff += ((gx - gy) ** 2).sum()
            # grad_diff = grad_diff / grad_count * 1000
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
    pyplot.imshow(ch.squeeze(input * 255).cpu().detach().permute(1, 2, 0).numpy().astype('uint8'))
    pyplot.savefig('ori.png')
    pyplot.imshow(ch.squeeze(dummy_data.detach() * 255).cpu().detach().permute(1, 2, 0).numpy().astype('uint8'))
    pyplot.savefig('atk.png')
    return dummy_data


def attack_iDlg(input, model, target):
    criterion = nn.CrossEntropyLoss()
    input = input.requires_grad_(True)
    pred, _, _ = model.forward(input, target)
    target = target.view(1, )
    y = criterion(pred, target)
    dy_dx = ch.autograd.grad(y, model.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    dummy_data = ch.randn(input.size()).to("cuda:0").requires_grad_(True).to(ch.float)
    optimizer = LBFGS([dummy_data, ], lr=0.05)
    label_pred = ch.argmin(ch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(
        False)
    for iters in range(500):
        def closure():
            optimizer.zero_grad()

            pred, _, _ = model.forward(dummy_data, target)
            dummy_loss = criterion(pred,
                                   label_pred)  # TODO: fix the gt_label to dummy_label in both code and slides.
            dummy_dy_dx = ch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):  # TODO: fix the variablas here
                grad_diff += ((gx - gy) ** 2).sum()
            # grad_diff = grad_diff / grad_count * 1000
            grad_diff.backward()

            return grad_diff

        optimizer.step(closure)
    pyplot.imshow(ch.squeeze(input * 255).cpu().detach().permute(1, 2, 0).numpy().astype('uint8'))
    pyplot.savefig('ori.png')
    pyplot.imshow(ch.squeeze(dummy_data.detach() * 255).cpu().detach().permute(1, 2, 0).numpy().astype('uint8'))
    pyplot.savefig('atk.png')
    return dummy_data


def train_decoder(train_loader, models, itr=10, lr=0.005):
    decoder = dec_model().to('cuda:0')
    print("\n Training Decoder \n")
    for i in range(itr):
        input, est = decoder_loop(decoder, train_loader, models, i, lr)
    pyplot.imshow(ch.squeeze(input * 255).cpu().detach().permute(1, 2, 0).numpy().astype('uint8'))
    pyplot.savefig('/output/ori.png')
    pyplot.imshow(ch.squeeze(est.detach() * 255).cpu().detach().permute(1, 2, 0).numpy().astype('uint8'))
    pyplot.savefig('/output/atk.png')


def decoder_loop(decoder, train_loader, models, i, lr=0.005):
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    dec_opt = Adam(decoder.decoder.parameters(), lr)
    loss_enc = AverageMeter()

    for i, (input, target) in iterator:
        rep_out = models.encoder.forward(input)
        # rep_out = add_Noise(rep_out, target, models)
        est, loss = decoder(input, rep_out)
        dec_opt.zero_grad()
        loss.backward()
        dec_opt.step()
        loss_enc.update(loss.item(), input.size(0))
        desc = ('Epoch:{0} | '
                'Loss {Loss:.4f} | '
        .format(
            i,
            Loss=loss_enc.avg))
        iterator.set_description(desc)
        iterator.refresh()
    return input[0], est[0]



def label_to_onehot(target, num_classes=10):
    target = ch.unsqueeze(target, 1)
    onehot_target = ch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return ch.mean(ch.sum(- target * F.log_softmax(pred, dim=-1), 1))
