import torch
from torchvision import models
import os


def __load_state_dict(model, name):
    if os.path.exists(name):
        model.load_state_dict(torch.load(name, map_location=torch.device('cpu')))
    return model


def get_board_model(name='board.pth', thresh=0.05):
    model = models.detection.fcos_resnet50_fpn(num_classes=4 + 1,
                                               detections_per_img=8,
                                               score_thresh=thresh,
                                               weights_backbone=None)
    return __load_state_dict(model, name)


def get_stone_model(name='stone.pth'):
    model = models.efficientnet_b3(num_classes=6)
    return __load_state_dict(model, name)


def get_part_board_model(name='part_board.pth', thresh=0.3):
    # failed, it's not working
    model = models.detection.fcos_resnet50_fpn(num_classes=19 * 19 + 1,
                                               detections_per_img=18 * 18,
                                               score_thresh=thresh,
                                               weights_backbone=None)
    return __load_state_dict(model, name)


def get_board_mobile_model(name='board_mobile.pth', thresh=0.05):
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(num_classes=4 + 1,
                                                               detections_per_img=8,
                                                               score_thresh=thresh,
                                                               weights_backbone=None
                                                               )
    return __load_state_dict(model, name)
