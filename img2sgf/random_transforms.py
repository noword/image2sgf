import torch
import pathlib
from typing import Optional, Dict, Sequence
import random
import torchvision.transforms as T
import torchvision.transforms.functional as F
from .transforms import RandomPhotometricDistort, Compose, ToTensor
import cv2
import numpy as np
from PIL import Image, ImageDraw
from .misc import get_4points_from_box


def perspective_target_with_matrix(target, mat):
    if 'keypoints' in target:
        keypoints = target['keypoints'].numpy()
        keypoints[:, :, :2] = cv2.perspectiveTransform(keypoints[:, :, :2], mat)
        target['keypoints'] = torch.as_tensor(keypoints)

    if 'boxes' in target:
        for i, box in enumerate(target['boxes']):
            x0, y0, x1, y1 = box.tolist()
            points = np.array([get_4points_from_box(x0, y0, x1, y1)])
            points = cv2.perspectiveTransform(points, mat)
            x = points[:, :, 0]
            y = points[:, :, 1]
            min_x = np.min(x)
            max_x = np.max(x)
            min_y = np.min(y)
            max_y = np.max(y)
            target['boxes'][i] = torch.as_tensor([min_x, min_y, max_x, max_y], dtype=torch.float32)

    if 'area' in target:
        boxes = target['boxes']
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    return target


def perspective_target(target: Optional[Dict[str, torch.Tensor]], startpoints: Sequence, endpoints: Sequence):
    mat = cv2.getPerspectiveTransform(np.array(startpoints, np.float32), np.array(endpoints, np.float32))
    return perspective_target_with_matrix(target, mat)


class RandomBackground(torch.nn.Module):
    def __init__(self,
                 dir_path='background',
                 img_size_range=(512, 768),
                 bg_size_range=(896, 1536),
                 rotation_degrees=(-30, 30),
                 donoting_probability=0.1,
                 perspectiv_probability=0.8
                 ):
        super().__init__()
        self.img_size_min, self.img_size_max = img_size_range
        self.bg_size_min, self.bg_size_max = bg_size_range
        self.degree_min, self.degree_max = rotation_degrees
        self.donoting_probability = donoting_probability
        self.perspectiv_probability = perspectiv_probability
        self.bgs = []
        for path in pathlib.Path(dir_path).glob('*.jpg'):
            self.bgs.append(str(path))

    def _resize(self, img):
        org_h, org_w = img.shape[-2:]
        img_w = img_h = random.randint(self.img_size_min, self.img_size_max)
        img = F.resize(img, (img_h, img_w))
        mat = cv2.getPerspectiveTransform(np.array(get_4points_from_box(0, 0, org_w, org_h), np.float32),
                                          np.array(get_4points_from_box(0, 0, img_w, img_h), np.float32))
        return img, mat

    def _perspective(self, img):
        org_h, org_w = img.shape[-2:]
        startpoints, endpoints = T.RandomPerspective.get_params(org_w, org_h, 0.4)
        img = F.perspective(img, startpoints, endpoints)
        mat = cv2.getPerspectiveTransform(np.array(startpoints, np.float32),
                                          np.array(endpoints, np.float32))
        return img, mat, endpoints

    def _add_background(self, img, endpoints):
        idx = random.randint(1, len(self.bgs) - 1)

        org_h, org_w = img.shape[-2:]
        bg_w = random.randint(org_w, self.bg_size_max)
        bg_h = random.randint(org_h, self.bg_size_max)
        x = random.randint(0, bg_w - org_w)
        y = random.randint(0, bg_h - org_h)
        bg_img = Image.open(self.bgs[idx]).resize((bg_w, bg_h))
        _img = T.ToPILImage()(img)

        if endpoints is None:
            bg_img.paste(_img, (x, y))
        else:
            alpha = Image.new('L', _img.size)
            draw = ImageDraw.ImageDraw(alpha)
            draw.polygon(tuple([tuple(x) for x in endpoints]), 'white')
            bg_img.paste(_img, (x, y), mask=alpha)

        mat = cv2.getPerspectiveTransform(np.array(get_4points_from_box(0, 0, 100, 100), np.float32),
                                          np.array(get_4points_from_box(x, y, x + 100, y + 100), np.float32))

        return T.ToTensor()(bg_img), mat

    def _rotate(self, img):
        angle = random.randint(self.degree_min, self.degree_max)
        oh, ow = img.shape[1:]
        img = F.rotate(img, angle, expand=True)

        h, w = img.shape[1:]
        mat = cv2.getRotationMatrix2D((ow / 2, oh / 2), angle, 1.0)
        mat[0, 2] += (w - ow) / 2
        mat[1, 2] += (h - oh) / 2
        mat = np.append(mat, [[0, 0, 1]], axis=0)

        return img, mat

    def forward(self, img, target: Optional[Dict[str, torch.Tensor]]):
        if random.random() < self.donoting_probability:
            return img, target

        endpoints = None
        # resize
        img, mat = self._resize(img)

        if random.random() < self.perspectiv_probability:
            # perspective
            img, _mat, endpoints = self._perspective(img)
            mat = _mat.dot(mat)

        # bg image
        img, _mat = self._add_background(img, endpoints=endpoints)
        mat = _mat.dot(mat)

        # rotate
        img, _mat = self._rotate(img)
        mat = _mat.dot(mat)

        # adjust target
        target = perspective_target_with_matrix(target, mat)

        return img, target


class GaussianBlur(T.GaussianBlur):
    def forward(self, img, target):
        return super(GaussianBlur, self).forward(img), target


class RandomCrop(T.RandomCrop):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__(size, padding, pad_if_needed, fill, padding_mode)

    def forward(self, img, target):
        return super().forward(img), target


class RandomNoise(torch.nn.Module):
    def __init__(self, p=0.5, mean=0.1, std=0.1):
        super().__init__()
        self.p = p
        self.mean = mean
        self.std = std

    def forward(self, img, target):
        if torch.rand(1) < self.p:
            img = img + torch.randn(img.shape) * self.std + self.mean
            img[img > 1.] = 1.
        return img, target


class RandomRectBrightness(torch.nn.Module):
    def __init__(self, p=0.5, brightness=(1., 2.)):
        super().__init__()
        self.p = p
        self._brightness = T.ColorJitter(brightness=brightness)

    def forward(self, img, target):
        if random.random() < self.p:
            c, h, w = img.shape
            _h = random.randint(0, h)
            h0, h1 = (0, _h) if bool(random.getrandbits(1)) else (_h, h)
            _w = random.randint(0, int(h * 0.8))
            w0, w1 = (0, _w) if bool(random.getrandbits(1)) else (_w, w)
            img[:, h0:h1, w0:w1] = self._brightness(img[:, h0:h1, w0:w1])
        return img, target


class ColorJitter(T.ColorJitter):
    def __init__(self, brightness=(0.875, 1.125), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.05, 0.05)):
        super().__init__(brightness, contrast, saturation, hue)

    def forward(self, img, target):
        return super().forward(img), target


class RandomRotation(T.RandomRotation):
    def __init__(self, **kwargs):
        super().__init__(expand=True, **kwargs)

    def forward(self, img, target):
        angle = self.get_params(self.degrees)
        oh, ow = img.shape[1:]
        img = F.rotate(img, angle, self.resample, self.expand, self.center, self.fill)

        h, w = img.shape[1:]
        mat = cv2.getRotationMatrix2D((ow / 2, oh / 2), angle, 1.0)
        mat[0, 2] += (w - ow) / 2
        mat[1, 2] += (h - oh) / 2
        mat = np.append(mat, [[0, 0, 1]], axis=0)

        target = perspective_target_with_matrix(target, mat)

        return img, target


def get_transform(train=False):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomNoise())
        transforms.append(GaussianBlur((3, 9)))
        transforms.append(RandomRectBrightness(p=.8))
        transforms.append(RandomBackground())
        transforms.append(RandomPhotometricDistort())
    return Compose(transforms)


def get_stone_transform(train=False):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomNoise())
        transforms.append(RandomPhotometricDistort(brightness=(0.875, 2.)))
        transforms.append(GaussianBlur((3, 9)))
        transforms.append(RandomCrop(42))
    return Compose(transforms)
