from model import get_board_model, get_stone_model, get_transform
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision
from misc import NpBoxPostion
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
from PIL import Image


DEFAULT_IMAGE_SIZE = 1024


def get_board_image(pil_image: Image):
    img = T.ToTensor()(pil_image)
    target = board_model(img.unsqueeze(0))[0]
    print(target)
    nms = torchvision.ops.nms(target['boxes'], target['scores'], 0.1)
    _boxes = target['boxes'].detach()[nms]
    _labels = target['labels'].detach()[nms]
    assert len(set(_labels)) >= 4

    boxes = [None] * 4
    for i, box in enumerate(_boxes):
        label = _labels[i] - 1
        if boxes[label] is None:
            boxes[label] = box.numpy()

    boxes = np.array(boxes)
    print(boxes)

    box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
    startpoints = boxes[:, :2].tolist()
    endpoints = [box_pos[18][0][:2],  # top left
                 box_pos[18][18][:2],  # top right
                 box_pos[0][0][:2],  # bottom left
                 box_pos[0][18][:2]  # bottom right
                 ]

    transform = cv2.getPerspectiveTransform(np.array(startpoints, np.float32), np.array(endpoints, np.float32))
    _img = cv2.warpPerspective(np.array(pil_image), transform, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

    return _img, boxes


def classifier_board(image: np.array):
    board = [[None] * 19 for _ in range(19)]
    box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
    img = T.ToTensor()(image)
    for x in range(19):
        for y in range(19):
            x0, y0, x1, y1 = box_pos[x][y].astype(int)
            result = stone_model(img[:, y0:y1, x0:x1].unsqueeze(0))[0]
            board[x][y] = result.argmax()
    return board


def demo(pil_img):
    _img, boxes = get_board_image(pil_img)
    board = classifier_board(_img)

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3)
    ax0.imshow(pil_img)
    for box in boxes:
        ax0.add_patch(Rectangle((box[0], box[1]),
                                box[2] - box[0],
                                box[3] - box[1],
                                linewidth=1,
                                edgecolor='g',
                                facecolor='none'))

    ax1.imshow(_img)
    box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
    for _boxes in box_pos:
        for box in _boxes:
            ax1.add_patch(Rectangle(box[:2],
                                    box_pos.grid_size,
                                    box_pos.grid_size,
                                    linewidth=0.5,
                                    edgecolor='b',
                                    facecolor='none'
                                    ))
    ax2.imshow(_img)
    for y in range(19):
        for x in range(19):
            label = board[x][y]
            color = label >> 1
            if color > 0:
                _x, _y = box_pos._grid_pos[x][y]
                ax2.plot(_x, _y, 'go' if color == 1 else 'b^', mfc='none')

    plt.show()


if __name__ == '__main__':
    board_model = get_board_model(thresh=0.5)
    board_model.load_state_dict(torch.load('weiqi_board.pth', map_location=torch.device('cpu')))
    board_model.eval()

    stone_model = get_stone_model()
    stone_model.load_state_dict(torch.load('weiqi_stone.pth', map_location=torch.device('cpu')))
    stone_model.eval()

    demo(Image.open(sys.argv[1]))
