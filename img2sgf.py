from model import get_board_model, get_stone_model, get_board_model_resnet50
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
from datetime import datetime
import argparse


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
    box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
    img = T.ToTensor()(image)
    imgs = torch.empty((19 * 19, 3, int(box_pos.grid_size), int(box_pos.grid_size)))

    for x in range(19):
        for y in range(19):
            x0, y0, x1, y1 = box_pos[x][y].astype(int)
            imgs[x + y * 19] = img[:, y0:y1, x0:x1]
    results = stone_model(imgs)

    board = [[None] * 19 for _ in range(19)]
    for x in range(19):
        for y in range(19):
            board[x][y] = results[x + y * 19].argmax()

    return board


def demo(img_name):
    pil_img = Image.open(img_name)
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
            color = board[x][y] >> 1
            if color > 0:
                _x, _y = box_pos._grid_pos[x][y]
                ax2.plot(_x, _y, 'go' if color == 1 else 'b^', mfc='none')

    plt.show()


S = 'abcdefghijklmnopqrs'


def img2sgf(img_name, sgf_name):
    _img, boxes = get_board_image(Image.open(img_name))
    board = classifier_board(_img)
    blacks = []
    whites = []
    for y in range(19):
        for x in range(19):
            color = board[x][y] >> 1
            if color == 1:
                blacks.append(f'[{S[y]}{S[18-x]}]')
            elif color == 2:
                whites.append(f'[{S[y]}{S[18-x]}]')

    with open(sgf_name, 'w') as fp:
        fp.write(f'(;GM[1]FF[4]CA[UTF-8]AP[img2sgf]KM[7.5]SZ[19]DT[{datetime.now().strftime("%Y-%m-%d")}]')
        fp.write('AB' + ''.join(blacks))
        fp.write('AW' + ''.join(whites))


def get_models():
    # board_model = get_board_model_resnet50(thresh=0.5)
    # board_model.load_state_dict(torch.load('weiqi_board_resnet50.pth', map_location=torch.device('cpu')))
    board_model = get_board_model(thresh=0.5)
    board_model.load_state_dict(torch.load('weiqi_board.pth', map_location=torch.device('cpu')))
    board_model.eval()

    stone_model = get_stone_model()
    stone_model.load_state_dict(torch.load('weiqi_stone.pth', map_location=torch.device('cpu')))
    stone_model.eval()

    return board_model, stone_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_name', action='store', nargs='?', help='input image file name')
    parser.add_argument('sgf_name', action='store', nargs='?', help='output sgf file name')
    parser.add_argument('--demo', action='store_true', default=False, help='show the action')
    args = parser.parse_args()

    board_model, stone_model = get_models()

    if args.image_name:
        if args.sgf_name:
            img2sgf(args.image_name, args.sgf_name)

        if args.demo:
            demo(args.image_name)
    else:
        parser.print_help()
