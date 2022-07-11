from PIL import Image
import torchvision
from .model import get_board_model, get_stone_model
from .misc import NpBoxPostion
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch
import os
from sgfmill import sgf
from datetime import datetime
import cv2

DEFAULT_IMAGE_SIZE = 1024


def get_models(board_path='board.pth', stone_path='stone.pth'):
    # board_model = get_board_model_resnet50(thresh=0.5)
    # board_model.load_state_dict(torch.load('weiqi_board_resnet50.pth', map_location=torch.device('cpu')))
    board_model = get_board_model(thresh=0.4)
    board_model.load_state_dict(torch.load(board_path, map_location=torch.device('cpu')))
    board_model.eval()

    stone_model = get_stone_model()
    stone_model.load_state_dict(torch.load(stone_path, map_location=torch.device('cpu')))
    stone_model.eval()

    return board_model, stone_model


def get_board_image(board_model, pil_image: Image):
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
            # print(int(label), float(_scores[i]), boxes[label])

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

    return Image.fromarray(_img), boxes, scores


def classifier_board(stone_model, image, save_images=False):
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
    # print(results.flip(0))

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


def get_sgf(board):
    blacks = []
    whites = []
    for y in range(19):
        for x in range(19):
            color = board[x][y] >> 1
            if color == 1:
                blacks.append([x, y])
            elif color == 2:
                whites.append([x, y])

    game = sgf.Sgf_game(size=19)
    game.set_date(datetime.now())
    root_node = game.get_root()
    root_node.set('AP', ('img2sgf', '1.0'))
    root_node.set_setup_stones(blacks, whites)
    return game
