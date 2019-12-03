import numpy as np
import cv2, torch
import os
from PIL import Image
import scipy.ndimage as ndi


class DiceLoss(torch.autograd.Function):
    # Dice Loss implementation for segmentation model
    # Code from: https://github.com/mattmacy/torchbiomed/blob/master/torchbiomed/loss.py
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, input, target, save=True):
        print("save: ", save)
        if save:
            print("Input/Target Shapes: ", input.shape, target.shape)
            self.save_for_backward(input, target)
        eps = 0.000001
        _, result_ = input.max(1)  # Argmax along channel dimension
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
        intersect = torch.sum(torch.mul(result, target))
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum + (2*eps)

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        IoU = intersect / union
        # print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} IoU: result_sum: {:.0f} IoU {:.7f}'.format(
        #     union, intersect, target_sum, result_sum, 2*IoU))
        out = torch.FloatTensor(1).fill_(2*IoU)
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        gt = torch.div(target, union)
        IoU2 = intersect/(union*union)
        pred = torch.mul(input[:, 1], IoU2)
        dDice = torch.add(torch.mul(gt, 2), torch.mul(pred, -4))
        grad_input = torch.cat((torch.mul(dDice, -grad_output[0]),
                                torch.mul(dDice, grad_output[0])), 0)
        return grad_input, None


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def normalize(a):
    a -= a.min()
    a /= a.max()
    return a

def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    return grad_y, grad_x

def imgrad_yx(img):
    N,C,_,_ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)

def reg_scalor(grad_yx):
    return torch.exp(-torch.abs(grad_yx)/255.)

def get_boundary_map(segmap):
    bitmap = np.zeros_like(segmap)
    # VD: Update code for cv2 version >= 4.0. Method now mutates input array, doesn't return a copy of it
    im2 = np.asarray(segmap).copy()
    # print("findContours Debugging Info: ", type(im2), im2.shape)
    contours, hierarchy = cv2.findContours(im2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bitmap = cv2.drawContours(bitmap, contours, -1, 1, 1) * 255
    return Image.fromarray(np.uint8(bitmap))

def distance_transform(mask, clip_background_distance=True, normalized=True):
    mask = np.asarray(mask)
    invalid = mask < 0. # {-1, 0, 1} pixel values
    foreground = mask.copy()
    background = 1. - foreground
    foreground[invalid] = 0.
    background[invalid] = 0.

    foreground_dist = ndi.distance_transform_edt(foreground)
    background_dist = ndi.distance_transform_edt(background)

    if clip_background_distance:
        foreground_max = foreground_dist.max()
        background_dist[background_dist > foreground_max] = foreground_max

    distance = foreground_dist * foreground + background_dist * background

    if normalized:
        distance = normalize(distance)

    return Image.fromarray(distance)
