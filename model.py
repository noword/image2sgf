from torchvision import models
from random_transforms import *
import transforms as T


def get_transform(train=False):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(RandomNoise())
        transforms.append(GaussianBlur((3, 9)))
        transforms.append(RandomRectBrightness(p=.8))
        transforms.append(RandomBackground())
        transforms.append(T.RandomPhotometricDistort())
    return T.Compose(transforms)


def get_stone_transform(train=False):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(RandomNoise())
        transforms.append(T.RandomPhotometricDistort(brightness=(0.875, 2.)))
        transforms.append(GaussianBlur((3, 9)))
        transforms.append(RandomCrop(42))
    return T.Compose(transforms)

# deprecated
# def get_model(thresh=0.05):
#     return models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
#                                    num_classes=19 * 19 * 2 + 4 + 1,
#                                    box_detections_per_img=int(19 * 19 * 1.2),
#                                    box_score_thresh=thresh)


def get_board_model_resnet50(thresh=0.05):
    return models.detection.fasterrcnn_resnet50_fpn(pretrained=False,
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
    # return models.mobilenet_v3_small(pretrained=False, num_classes=6)
    # return models.mobilenet_v3_large(pretrained=False, num_classes=6)
    return models.efficientnet_b0(pretrained=False, num_classes=6)


if __name__ == '__main__':
    print(get_board_model())
