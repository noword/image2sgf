from torchvision import models


def get_board_model(thresh=0.05):
    return models.detection.fcos_resnet50_fpn(num_classes=4 + 1,
                                              detections_per_img=8,
                                              score_thresh=thresh)


def get_stone_model():
    return models.efficientnet_b0(num_classes=6)


if __name__ == '__main__':
    print(get_board_model())
