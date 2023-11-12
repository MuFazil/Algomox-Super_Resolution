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
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms.functional import rgb_to_grayscale
import re

# from torch.utils.tensorboard import SummaryWriter

# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio


def train(args):
    # writer = SummaryWriter()

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

    generator = Generator(
        img_feat=3, n_feats=128, kernel_size=3, num_block=args.res_num, scale=args.scale
    )

    if args.fine_tuning:
        generator.load_state_dict(torch.load(args.generator_path))
        print("pre-trained model is loaded")
        print("path : %s" % (args.generator_path))

    generator = generator.to(device)
    generator.train()

    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=1e-4)

    pre_epoch = 0
    fine_epoch = 0
    psnr_list = []  # List to store PSNR for each image

    #### Train using L2_loss
    while pre_epoch < args.pre_train_epoch:
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

            # writer.add_scalar("Loss/train", loss.item(), pre_epoch)
            # writer.add_scalar("PSNR/train", psnr, pre_epoch)

            # Log images to TensorBoard
            # if pre_epoch % 32 == 0:
            #     # Assuming you have a function to generate and return a sample image
            #     generated_image = generate_sample_image(generator, lr_sample)
            #     writer.add_image('Generated Image', generated_image, epoch)

            g_optim.zero_grad()
            loss.backward()
            g_optim.step()

        pre_epoch += 1

        if pre_epoch % 2 == 0:
            print(pre_epoch)
            print(loss.item())
            print("=========")

        if pre_epoch % 100 == 0:
            torch.save(
                generator.state_dict(), "./model/pre_trained_model_%03d.pt" % pre_epoch
            )

    # Calculate the average PSNR after the pre-training phase
    avg_psnr_pretrain = np.mean(psnr_list)
    print("Average PSNR after pre-training:", avg_psnr_pretrain)

    #### Train using perceptual & adversarial loss
    vgg_net = vgg19().to(device)
    vgg_net = vgg_net.eval()

    discriminator = Discriminator(patch_size=args.patch_size * args.scale)
    discriminator = discriminator.to(device)
    discriminator.train()

    d_optim = optim.Adam(discriminator.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

    VGG_loss = perceptual_loss(vgg_net)
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()
    real_label = torch.ones((args.batch_size, 1)).to(device)
    fake_label = torch.zeros((args.batch_size, 1)).to(device)

    # writer.close()

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

            g_optim.zero_grad()
            d_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            # writer.add_scalar("Loss/Fine", loss.item(), fine_epoch)
            # writer.add_scalar("Loss/Fine", g_loss, fine_epoch)
            # writer.add_scalar("Loss/Fine", percep_loss, fine_epoch)
            # writer.add_scalar("Loss/Fine", adversarial_loss, fine_epoch)
            # writer.add_scalar("PSNR/Fine", L2_loss, fine_epoch)

        fine_epoch += 1

        if fine_epoch % 2 == 0:
            print(fine_epoch)
            print(g_loss.item())
            print(d_loss.item())
            print("=========")

        if fine_epoch % 500 == 0:
            # torch.save(generator.state_dict(), './model/SRGAN_gene_%03d.pt'%fine_epoch)
            # torch.save(discriminator.state_dict(), './model/SRGAN_discrim_%03d.pt'%fine_epoch)
            torch.save(
                generator.state_dict(), "./model/SRGAN_gene_%03d.pt" % fine_epoch
            )
            torch.save(
                discriminator.state_dict(), "./model/SRGAN_discrim_%03d.pt" % fine_epoch
            )


# In[ ]:


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = mydata(
        GT_path=args.GT_path, LR_path=args.LR_path, in_memory=False, transform=None
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    generator = Generator(
        img_feat=3, n_feats=128, kernel_size=3, num_block=args.res_num
    )
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()

    # Remove the last 3 characters (.pt) from the model name
    model_name = args.generator_path[:-3]

    # Extract the last 4 digits from the modified model name
    model_number = model_name[-4:]

    # f = open("./results.txt", "w")
    # result_file_name = f"./results_{model_number}.txt"
    result_file_name = f"./validation_results/results_{model_number}.txt"

    result_folder = f"./validation_results"
    os.makedirs(result_folder, exist_ok=True)

    f = open(result_file_name, "w")

    psnr_list = []
    ssim_list = []

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

            # win_size = min(7, min(h, w) - 1)
            # Calculate SSIM
            # ssim_value = ssim(y_output, y_gt, data_range=1.0, multichannel=True)
            # ssim_list.append(ssim_value)
            # f.write(f"ssim : {ssim_value:.4f} \n")

            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save("./result/res_%04d.png" % i)

        f.write("avg psnr : %04f" % np.mean(psnr_list))
        # f.write(f"avg ssim : {np.mean(ssim_list):.4f}\n")


def test_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    dataset = testOnly_data(LR_path=args.LR_path, in_memory=False, transform=None)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )

    generator = Generator(
        img_feat=3, n_feats=128, kernel_size=3, num_block=args.res_num
    )
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
            result.save("./TestResults/res_%04d.png" % i)