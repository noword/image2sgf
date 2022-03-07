from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch import nn
import torchvision.transforms.functional as F
from random_background import RandomBackground
import transforms as T


class ToTensor(nn.Module):
    def forward(self, image, target):
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target


def get_transform(train=False):
    transforms = []
    transforms.append(ToTensor())
    transforms.append(RandomBackground())
    return T.Compose(transforms)


def get_model(thresh=0.05):
    return fasterrcnn_resnet50_fpn(pretrained=False,
                                   num_classes=19 * 19 * 2 + 4 + 1,
                                   box_detections_per_img=int(19 * 19 * 1.2),
                                   box_score_thresh=thresh)


def get_board_model(thresh=0.05):
    return fasterrcnn_resnet50_fpn(pretrained=False,
                                   num_classes=4 + 1,
                                   box_detections_per_img=8,
                                   box_score_thresh=thresh)


if __name__ == '__main__':
    print(get_model())
