from torchvision import models


def get_board_model(thresh=0.05):
    # return models.detection.fasterrcnn_resnet50_fpn_v2(num_classes=4 + 1,
    #                                                    detections_per_img=8,
    #                                                    score_thresh=thresh)
    return models.detection.fcos_resnet50_fpn(num_classes=4 + 1,
                                              detections_per_img=8,
                                              score_thresh=thresh,
                                              weights_backbone=None)


def get_stone_model():
    # return models.efficientnet_b0(num_classes=6)
    return models.efficientnet_b3(num_classes=6)


def get_part_board_model(thresh=0.05):
    return models.detection.fcos_resnet50_fpn(num_classes=19 * 19 + 1,
                                              detections_per_img=18 * 18,
                                              score_thresh=thresh,
                                              weights_backbone=None)


if __name__ == '__main__':
    print(get_board_model())
