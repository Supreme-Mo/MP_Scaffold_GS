#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
# 修改
def cos_loss(output, gt):

    dot = torch.sum(output * gt, dim=(0))
    norm_A = torch.linalg.norm(output, dim=(0))
    norm_B = torch.linalg.norm(gt, dim=(0))
    cos_sim = dot / (norm_A * norm_B+1e-8)

    return 1-torch.mean(cos_sim)

def gradient_x(img,gri_s):
    return img[:, :, :-gri_s] - img[:, :, gri_s:]

def gradient_y(img,gri_s):
    return img[:, :-gri_s] - img[:, gri_s:]

def get_normal_smoothness(depth, img,k_size=5):
    _,H,W = img.shape

    img_gradients_x = gradient_x(img,gri_s=int(k_size/2))
    img_gradients_y = gradient_y(img,gri_s=int(k_size/2))
    if k_size == 5:
        img_gradients_x = F.pad(img_gradients_x, (1,1, 0, 0))
        img_gradients_y = F.pad(img_gradients_y, (0, 0, 1,1))
    elif k_size == 3:
        img_gradients_x = F.pad(img_gradients_x, (0, 1, 0, 0))
        img_gradients_y = F.pad(img_gradients_y, (0, 0, 0, 1))

    gradient_weight = 0.5*img_gradients_y+0.5*img_gradients_x

    gradient_weight = torch.exp(-torch.mean(torch.abs(gradient_weight),dim=0,keepdim=True))
    gradient_weight = F.pad(gradient_weight, (0, H%k_size, 0, 0))
    gradient_weight = F.pad(gradient_weight, (0, 0, 0, W%k_size))

    # print(gradient_weight.shape)
    gradient_chunks = F.unfold(gradient_weight.unsqueeze(0), kernel_size=(k_size, k_size), stride=k_size) #(1,9,size)
    gradient_chunks = torch.mean(gradient_chunks,dim=1,keepdim=True)

    depth = F.pad(depth, (0, H%k_size, 0, 0))
    depth = F.pad(depth, (0, 0, 0, W%k_size))

    depth_chunks = F.unfold(depth.unsqueeze(0), kernel_size=(k_size, k_size), stride=k_size) #(1,9,size)
    depth_var = torch.var(depth_chunks,dim=1,keepdim=True) # (1,1,size)

    return torch.mean(gradient_chunks*depth_var)
