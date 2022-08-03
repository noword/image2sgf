from PIL import Image
import torchvision
from .model import get_board_model, get_stone_model, get_part_board_model
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


def get_models(board_path='board.pth', part_board_path='part_board.pth', stone_path='stone.pth'):
    board_model = stone_model = part_board_model = None

    if os.path.exists(board_path):
        board_model = get_board_model(thresh=0.4)
        board_model.load_state_dict(torch.load(board_path, map_location=torch.device('cpu')))
        board_model.eval()

    if os.path.exists(stone_path):
        stone_model = get_stone_model()
        stone_model.load_state_dict(torch.load(stone_path, map_location=torch.device('cpu')))
        stone_model.eval()

    if os.path.exists(part_board_path):
        part_board_model = get_part_board_model()
        part_board_model.load_state_dict(torch.load(part_board_path, map_location=torch.device('cpu')))
        part_board_model.eval()

    return board_model, part_board_model, stone_model


def expand_image(pil_image):
    w, h = pil_image.size

    # get a color for background
    colors = {}
    for y in range(0, h, 10):
        for x in range(0, w, 10):
            c = pil_image.getpixel((x, y))
            if c in colors:
                colors[c] += 1
            else:
                colors[c] = 1
    colors = [(k, v) for k, v in colors.items()]
    colors.sort(key=lambda x: x[1])
    c = colors[-1][0]

    width = max(w, h)
    if min(w, h) / width < 0.9:
        width = int(width * 1.2)

    img = Image.new('RGB', (width, width), c)
    left = (width - w) // 2
    top = (width - h) // 2
    img.paste(pil_image, (left, top))

    return img, left, top


def get_board_position(board_model, image, expand=True):
    # return 4 corners info
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    if image.mode != 'RGB':
        image = image.convert('RGB')

    if expand:
        img, x_offset, y_offset = expand_image(image)
    else:
        img = image
        x_offset = y_offset = 0

    target = board_model(T.ToTensor()(img).unsqueeze(0))[0]
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

    assert [0] * 4 not in boxes

    boxes[:, ::2] -= x_offset
    boxes[:, 1::2] -= y_offset

    return boxes, scores


def get_board_image(board_model, img, expand=True):
    boxes, scores = get_board_position(board_model, img, expand)

    box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
    startpoints = boxes[:, :2].tolist()
    endpoints = [box_pos[18][0][:2],  # top left
                 box_pos[18][18][:2],  # top right
                 box_pos[0][0][:2],  # bottom left
                 box_pos[0][18][:2]  # bottom right
                 ]

    transform = cv2.getPerspectiveTransform(np.array(startpoints, np.float32), np.array(endpoints, np.float32))
    _img = cv2.warpPerspective(np.array(img), transform, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

    return Image.fromarray(_img), boxes, scores


def classifer_part_board(part_board_model, stone_model, pil_image, save_images=False):
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    target = part_board_model(T.ToTensor()(pil_image).unsqueeze(0))[0]
    nms = torchvision.ops.nms(target['boxes'], target['scores'], 0.05)
    _boxes = target['boxes'].detach()[nms]
    _labels = target['labels'].detach()[nms]
    _scores = target['scores'].detach()[nms]

    imgs = torch.empty((len(_boxes), 3, 64, 64))
    for i, box in enumerate(_boxes.to(torch.int32)):
        img = pil_image.crop(box.tolist())
        img = img.resize((64, 64))
        imgs[i] = T.ToTensor()(img)

    results = stone_model(imgs).argmax(1)

    return _boxes, _labels, _scores, results


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
