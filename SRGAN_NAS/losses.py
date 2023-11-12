import torch
import torch.nn as nn
from torchvision import transforms


# This class extends nn.Conv2d and is used for mean shift operation.
# It shifts the mean of input images to match a target mean and standard deviation.
# The mean and standard deviation are set using the provided parameters norm_mean and norm_std.
# It ensures that the images are normalized consistently during the processing.
class MeanShift(nn.Conv2d):
    def __init__(
        self,
        rgb_range=1,
        norm_mean=(0.485, 0.456, 0.406),
        norm_std=(0.229, 0.224, 0.225),
        sign=-1,
    ):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(norm_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(norm_mean) / std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for p in self.parameters():
            p.requires_grad = False


# This class defines a perceptual loss module using a pre-trained VGG network.
# It includes a mean shift transformation and calculates the mean squared error loss between high-resolution (HR) and super-resolved (SR) features.
# The perceptual loss is computed at a specified layer of the VGG network, allowing for feature-wise loss evaluation
class perceptual_loss(nn.Module):
    def __init__(self, vgg):
        super(perceptual_loss, self).__init__()
        self.normalization_mean = [0.485, 0.456, 0.406]
        self.normalization_std = [0.229, 0.224, 0.225]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = MeanShift(
            norm_mean=self.normalization_mean, norm_std=self.normalization_std
        ).to(self.device)
        self.vgg = vgg
        self.criterion = nn.MSELoss()

    def forward(self, HR, SR, layer="relu5_4"):
        ## HR and SR should be normalized [0,1]
        hr = self.transform(HR)
        sr = self.transform(SR)

        hr_feat = getattr(self.vgg(hr), layer)
        sr_feat = getattr(self.vgg(sr), layer)

        return self.criterion(hr_feat, sr_feat), hr_feat, sr_feat


# The TVLoss class implements the Total Variation (TV) loss, which encourages spatial smoothness in the output.
# It calculates the TV loss based on the differences between neighboring pixel values in both horizontal and vertical directions.
# The weight of the TV loss can be adjusted using the tv_loss_weight parameter.
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()

        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
