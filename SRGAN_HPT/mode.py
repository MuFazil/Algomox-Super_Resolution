import nni
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss
from dataset import *
from srgan_model import Generator, Discriminator
from vgg19 import vgg19
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio


# Function to train the model
def train(args):
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([crop(args.scale, args.patch_size), augmentation()])
    dataset = mydata(
        GT_path=args.GT_path,
        LR_path=args.LR_path,
        in_memory=args.in_memory,
        transform=transform,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )

    # Initializing the generator model
    generator = Generator(
        img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num, scale=args.scale
    )

    if args.fine_tuning:
        generator.load_state_dict(torch.load(args.generator_path))
        print("pre-trained model is loaded")
        print("path : %s" % (args.generator_path))

    learning_rate = args.learning_rate
    batch_size = args.batch_size
    pre_train_epoch = args.pre_train_epoch

    generator = generator.to(device)
    generator.train()

    # Loss function and optimizer for generator
    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=learning_rate)

    pre_epoch = 0
    fine_epoch = 0

    psnr_list = []  # List to store PSNR for each image

    #### Train using L2_loss
    while pre_epoch < pre_train_epoch:
        for i, tr_data in enumerate(loader):
            gt = tr_data["GT"].to(device)
            lr = tr_data["LR"].to(device)

            output, _ = generator(lr)
            loss = l2_loss(gt, output)

            gt_np = (gt + 1.0).cpu().detach().numpy() / 2.0
            output_np = (output + 1.0).cpu().detach().numpy() / 2.0

            # Calculate PSNR for each image and add to the list
            psnr = peak_signal_noise_ratio(gt_np, output_np, data_range=1.0)
            psnr_list.append(psnr)

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()

        pre_epoch += 1

        if pre_epoch % 2 == 0:
            print(pre_epoch)
            print(loss.item())
            print("=========")

        if pre_epoch % 2000 == 0:
            # torch.save(
            #     generator.state_dict(), "./HPTmodel/optimized_model_%03d.pt" % pre_epoch
            # )
            filename = "./HPTmodel/optimized_model_bs{}_ep{}_te{}_lr{:.0e}.pt".format(
                batch_size, pre_epoch, pre_train_epoch, learning_rate
            )
            torch.save(generator.state_dict(), filename)

    # Calculate the average PSNR after the pre-training phase
    avg_psnr_pretrain = np.mean(psnr_list)
    print("Average PSNR after pre-training:", avg_psnr_pretrain)
    nni.report_final_result(avg_psnr_pretrain)

    #### Train using perceptual & adversarial loss
    vgg_net = vgg19().to(device)
    vgg_net = vgg_net.eval()

    discriminator = Discriminator(patch_size=args.patch_size * args.scale)
    discriminator = discriminator.to(device)
    discriminator.train()

    d_optim = optim.Adam(discriminator.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

    VGG_loss = perceptual_loss(vgg_net)
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()
    real_label = torch.ones((args.batch_size, 1)).to(device)
    fake_label = torch.zeros((args.batch_size, 1)).to(device)

    # Fine tuning
    while fine_epoch < args.fine_train_epoch:
        scheduler.step()

        for i, tr_data in enumerate(loader):
            gt = tr_data["GT"].to(device)
            lr = tr_data["LR"].to(device)

            ## Training Discriminator
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)

            d_loss_real = cross_ent(real_prob, real_label)
            d_loss_fake = cross_ent(fake_prob, fake_label)

            d_loss = d_loss_real + d_loss_fake

            g_optim.zero_grad()
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            ## Training Generator
            output, _ = generator(lr)
            fake_prob = discriminator(output)

            _percep_loss, hr_feat, sr_feat = VGG_loss(
                (gt + 1.0) / 2.0, (output + 1.0) / 2.0, layer=args.feat_layer
            )

            L2_loss = l2_loss(output, gt)
            percep_loss = args.vgg_rescale_coeff * _percep_loss
            adversarial_loss = args.adv_coeff * cross_ent(fake_prob, real_label)
            total_variance_loss = args.tv_loss_coeff * tv_loss(
                args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2
            )

            g_loss = percep_loss + adversarial_loss + total_variance_loss + L2_loss

            # Calculate PSNR for each image and add to the list
            gt_np = (gt + 1.0).cpu().detach().numpy() / 2.0
            output_np = (output + 1.0).cpu().detach().numpy() / 2.0

            # Calculate PSNR for each image and add to the list
            psnr = peak_signal_noise_ratio(gt_np, output_np, data_range=1.0)
            psnr_list.append(psnr)

            g_optim.zero_grad()
            d_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

        fine_epoch += 1

        if fine_epoch % 2 == 0:
            print(fine_epoch)
            print(g_loss.item())
            print(d_loss.item())
            print("=========")

        if fine_epoch % 500 == 0:
            torch.save(
                generator.state_dict(), "./model/SRGAN_gene_%03d.pt" % fine_epoch
            )
            torch.save(
                discriminator.state_dict(), "./model/SRGAN_discrim_%03d.pt" % fine_epoch
            )
    avg_psnr_finetune = np.mean(psnr_list)
    print("Average PSNR after fine tuning : ", (avg_psnr_finetune))


# In[ ]:

#Function to validate the model
def valid(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = mydata(
        GT_path=args.GT_path, LR_path=args.LR_path, in_memory=False, transform=None
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()

    f = open("./result.txt", "w")
    psnr_list = []

    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data["GT"].to(device)
            lr = te_data["LR"].to(device)

            bs, c, h, w = lr.size()
            gt = gt[:, :, : h * args.scale, : w * args.scale]

            output, _ = generator(lr)

            output = output[0].cpu().numpy()
            output = np.clip(output, -1.0, 1.0)
            gt = gt[0].cpu().numpy()

            output = (output + 1.0) / 2.0
            gt = (gt + 1.0) / 2.0

            output = output.transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)

            y_output = rgb2ycbcr(output)[
                args.scale : -args.scale, args.scale : -args.scale, :1
            ]
            y_gt = rgb2ycbcr(gt)[args.scale : -args.scale, args.scale : -args.scale, :1]

            psnr = peak_signal_noise_ratio(
                y_output / 255.0, y_gt / 255.0, data_range=1.0
            )

            # psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range = 1.0)
            psnr_list.append(psnr)
            f.write("psnr : %04f \n" % psnr)

            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save("./result/res_%04d.png" % i)

        f.write("avg psnr : %04f" % np.mean(psnr_list))


# Function to test the model
def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = testOnly_data(LR_path=args.LR_path, in_memory=False, transform=None)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()

    with torch.no_grad():
        for i, te_data in enumerate(loader):
            lr = te_data["LR"].to(device)
            output, _ = generator(lr)
            output = output[0].cpu().numpy()
            output = (output + 1.0) / 2.0
            output = output.transpose(1, 2, 0)
            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save("./result/res_%04d.png" % i)
