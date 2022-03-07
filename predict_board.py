from model import get_board_model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch
import sys
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision
from misc import BoxPostion
import cv2
import numpy as np
import os


def get_4corners(boxes):
    min_x = boxes[:, 0].min()
    min_y = boxes[:, 1].min()
    _sum = (boxes[:, 0] - min_x) + (boxes[:, 1] - min_y)
    _difference = (boxes[:, 0] - min_x) - (boxes[:, 1] - min_y)

    top_left = _sum.argmin()
    bottom_right = _sum.argmax()
    bottom_left = _difference.argmin()
    top_right = _difference.argmax()

    return boxes[[top_left, top_right, bottom_left, bottom_right], :]


def predict_demo(model, pil_image):
    img = T.ToTensor()(pil_image)
    target = model(img.unsqueeze(0))[0]
    print(target)

    nms = torchvision.ops.nms(target['boxes'], target['scores'], 0.1)
    boxes = target['boxes'].detach()[nms]
    boxes = get_4corners(boxes)

    fig, (ax0, ax1) = plt.subplots(ncols=2)

    ax0.imshow(pil_image)
    for box in boxes:
        ax0.add_patch(Rectangle((box[0], box[1]),
                                box[2] - box[0],
                                box[3] - box[1],
                                linewidth=1,
                                edgecolor='g',
                                facecolor='none'))

    size = 1024
    box_pos = BoxPostion(size, 19)
    startpoints = boxes[:, :2].tolist()
    endpoints = [box_pos[18][0][:2],  # top left
                 box_pos[18][18][:2],  # top right
                 box_pos[0][0][:2],  # bottom left
                 box_pos[0][18][:2]  # bottom right
                 ]

    transform = cv2.getPerspectiveTransform(np.array(startpoints, np.float32), np.array(endpoints, np.float32))
    _img = cv2.warpPerspective(np.array(pil_image), transform, (size, size))

    ax1.imshow(_img)
    ax1.add_patch(Rectangle((box_pos.x0, box_pos.y0),
                            box_pos.board_width,
                            box_pos.board_width,
                            linewidth=0.2,
                            edgecolor='g',
                            facecolor='none'))

    for boxes in box_pos:
        for box in boxes:
            ax1.add_patch(Rectangle(box[:2],
                                    box_pos.grid_size,
                                    box_pos.grid_size,
                                    linewidth=0.5,
                                    edgecolor='b',
                                    facecolor='none'
                                    ))

    plt.show()


if __name__ == '__main__':
    model = get_board_model()
    model.load_state_dict(torch.load('weiqi_board.pth', map_location=torch.device('cpu')))
    model.eval()

    if len(sys.argv) > 1:
        pil_image = Image.open(sys.argv[1])
        predict_demo(model, pil_image)
    else:
        count = 0
        while True:
            name = f'{count}.jpg'
            if not os.path.exists(name):
                break
            predict_demo(model, Image.open(name))
            count += 1
