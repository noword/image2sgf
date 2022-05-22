from img2sgf import *
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
from PIL import Image
from datetime import datetime
import argparse
import os
import time

DEFAULT_IMAGE_SIZE = 1024


def get_board_image(pil_image: Image):
    img = T.ToTensor()(pil_image)
    target = board_model(img.unsqueeze(0))[0]
    # print(target)
    nms = torchvision.ops.nms(target['boxes'], target['scores'], 0.1)
    _boxes = target['boxes'].detach()[nms]
    _labels = target['labels'].detach()[nms]
    _scores = target['scores'].detach()[nms]
    assert len(set(_labels)) >= 4

    boxes = np.zeros((4, 4))
    scores = [0] * 4
    for i, box in enumerate(_boxes):
        label = _labels[i] - 1
        if np.count_nonzero(boxes[label]) == 0:
            boxes[label] = box.numpy()
            scores[label] = float(_scores[i])
            print(int(label), float(_scores[i]), boxes[label])

    # print(boxes)
    assert [0] * 4 not in boxes

    box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
    startpoints = boxes[:, :2].tolist()
    endpoints = [box_pos[18][0][:2],  # top left
                 box_pos[18][18][:2],  # top right
                 box_pos[0][0][:2],  # bottom left
                 box_pos[0][18][:2]  # bottom right
                 ]

    transform = cv2.getPerspectiveTransform(np.array(startpoints, np.float32), np.array(endpoints, np.float32))
    _img = cv2.warpPerspective(np.array(pil_image), transform, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

    return _img, boxes, scores


def classifier_board(image: np.array, save_images=False):
    box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
    img = T.ToTensor()(image)
    imgs = torch.empty((19 * 19, 3, int(box_pos.grid_size), int(box_pos.grid_size)))

    for y in range(19):
        for x in range(19):
            x0, y0, x1, y1 = box_pos[y][x].astype(int)
            imgs[x + y * 19] = img[:, y0:y1, x0:x1]

    results = stone_model(imgs).argmax(1)

    if save_images:
        save_all_images(imgs, results)

    results = results.reshape(19, 19)
    print(results.flip(0))

    return results


def save_all_images(images, labels):
    path = 'stones'
    num_classes = 6
    counts = [0] * num_classes

    for i in range(num_classes):
        try:
            os.makedirs(f'{path}/{i}')
        except BaseException:
            pass
        while os.path.exists(f'{path}/{counts[i]}.jpg'):
            counts[i] += 1

    for i, img in enumerate(images):
        label = int(labels[i])
        T.ToPILImage()(img).save(f'{path}/{label}/{counts[label]}.jpg')
        counts[label] += 1


def demo(pil_img, save_images=False):
    if isinstance(pil_img, str):
        pil_img = Image.open(pil_img).convert('RGB')
    print('1st perspective')
    img0, boxes0, scores0 = get_board_image(pil_img)
    print('2nd perspective')
    img1, boxes1, scores1 = get_board_image(img0)
    board = classifier_board(img1, save_images)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, wspace=0.2, hspace=0.3)
    fig.canvas.manager.set_window_title('image to sgf demo')

    ax0.set_title('detect 4 corners')
    ax0.imshow(pil_img)
    for i, box in enumerate(boxes0):
        ax0.add_patch(Rectangle((box[0], box[1]),
                                box[2] - box[0],
                                box[3] - box[1],
                                linewidth=1,
                                edgecolor='g',
                                facecolor='none'))
        ax0.text(box[0], box[1], f'{scores0[i]:.2f}', color='r')

    ax1.set_title(f'perspective correct the board, then detect 4 corners again')
    ax1.imshow(img0)
    box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)

    for i, box in enumerate(boxes1):
        ax1.add_patch(Rectangle((box[0], box[1]),
                                box[2] - box[0],
                                box[3] - box[1],
                                linewidth=1,
                                edgecolor='g',
                                facecolor='none'))
        ax1.text(box[0], box[1], f'{scores1[i]:.2f}', color='r')

    ax2.set_title(f'perspective correct the board again')
    ax2.imshow(img1)
    box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
    for _boxes in box_pos:
        for box in _boxes:
            ax2.add_patch(Rectangle(box[:2],
                                    box_pos.grid_size,
                                    box_pos.grid_size,
                                    linewidth=0.5,
                                    edgecolor='b',
                                    facecolor='none'
                                    ))

    ax3.set_title('classify stones')
    ax3.imshow(img1)
    for y in range(19):
        for x in range(19):
            sign = board[x][y] & 1
            color = board[x][y] >> 1
            if color > 0:
                _x, _y = box_pos._grid_pos[x][y]
                ax3.plot(_x, _y, 'gs' if color == 1 else 'b^', mfc='none')  # if sign else None)

    plt.show()


S = 'abcdefghijklmnopqrs'


def img2sgf(img, sgf_name, save_images=False):
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    _img, _, scores = get_board_image(img)
    if min(scores) < 0.7:
        _img, _, _ = get_board_image(_img)

    board = classifier_board(_img, save_images)

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
        fp.write(')')


def get_models():
    # board_model = get_board_model_resnet50(thresh=0.5)
    # board_model.load_state_dict(torch.load('weiqi_board_resnet50.pth', map_location=torch.device('cpu')))
    board_model = get_board_model(thresh=0.4)
    board_model.load_state_dict(torch.load('board.pth', map_location=torch.device('cpu')))
    board_model.eval()

    stone_model = get_stone_model()
    stone_model.load_state_dict(torch.load('stone.pth', map_location=torch.device('cpu')))
    stone_model.eval()

    return board_model, stone_model


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('image_name', action='store', nargs='?', help='input image file name')
    parser.add_argument('--sgf_name', action='store', help='output sgf file name')
    parser.add_argument('--capture', action='store_true', default=False, help='capture the screenshot')
    parser.add_argument('--save_images', action='store_true', default=False, help='save grid images')
    args = parser.parse_args()

    board_model, stone_model = get_models()

    if args.capture or args.image_name is None:
        import pyautogui
        sleep_time = 2 - time.time() + start_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        img = pyautogui.screenshot()
    else:
        img = Image.open(args.image_name).convert("RGB")

    if args.sgf_name:
        img2sgf(img, args.sgf_name, args.save_images)
    else:
        demo(img, args.save_images)
