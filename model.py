from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch import nn
import torchvision.transforms as TT
import torchvision.transforms.functional as F
from random_background import RandomBackground
import transforms as T
from torchvision import models
import random
import torch


class GaussianBlur(TT.GaussianBlur):
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__(kernel_size, sigma)

    def forward(self, img, target):
        return super().forward(img), target


class RandomCrop(TT.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def forward(self, img, target):
        return super().forward(img), target


class RandomRectBrightness(nn.Module):
    def __init__(self, p=0.5, brightness=(1., 2.)):
        super().__init__()
        self.p = p
        self._brightness = TT.ColorJitter(brightness=brightness)

    def forward(self, img, target):
        if random.random() < self.p:
            c, h, w = img.shape
            _h = random.randint(0, h)
            h0, h1 = (0, _h) if bool(random.getrandbits(1)) else (_h, h)
            _w = random.randint(0, int(h * 0.8))
            w0, w1 = (0, _w) if bool(random.getrandbits(1)) else (_w, w)
            img[:, h0:h1, w0:w1] = self._brightness(img[:, h0:h1, w0:w1])
        return img, target


class ToTensor(nn.Module):
    def forward(self, image, target):
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target


def get_transform(train=False):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomRectBrightness(p=.8))
        transforms.append(RandomBackground())
        transforms.append(T.RandomPhotometricDistort())
        transforms.append(GaussianBlur((3, 9)))
    return T.Compose(transforms)


def get_stone_transform(train=False):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomPhotometricDistort(brightness=(0.875, 2.)))
        transforms.append(GaussianBlur((3, 9)))
        transforms.append(RandomCrop(42))
    return T.Compose(transforms)

# deprecated
# def get_model(thresh=0.05):
#     return fasterrcnn_resnet50_fpn(pretrained=False,
#                                    num_classes=19 * 19 * 2 + 4 + 1,
#                                    box_detections_per_img=int(19 * 19 * 1.2),
#                                    box_score_thresh=thresh)


def get_board_model_resnet50(thresh=0.05):
    return fasterrcnn_resnet50_fpn(pretrained=False,
                                   num_classes=4 + 1,
                                   box_detections_per_img=8,
                                   box_score_thresh=thresh)


def get_board_model(thresh=0.05):
    # return models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False,
    #                                                           num_classes=4 + 1,
    #                                                           box_detections_per_img=8,
    #                                                           box_score_thresh=thresh
    #                                                           )
    return models.detection.fcos_resnet50_fpn(pretrained=False,
                                              num_classes=4 + 1,
                                              detections_per_img=8,
                                              score_thresh=thresh)


def get_stone_model():
    return models.mobilenet_v3_small(pretrained=False, num_classes=6)


if __name__ == '__main__':
    print(get_board_model())
