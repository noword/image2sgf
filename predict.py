from model import get_model
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import torch
import sys
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision
from sgf2img import GridPosition
from misc import get_xy, S, get_4points_from_box
import cv2
import numpy as np

if __name__ == '__main__':
    pil_image = Image.open(sys.argv[1])
    img = T.ToTensor()(pil_image)

    model = get_model(thresh=0.9)
    model.load_state_dict(torch.load('weiqi.pth', map_location=torch.device('cpu')))
    model.eval()
    target = model(img.unsqueeze(0))[0]
    print(target)
    nms = torchvision.ops.nms(target['boxes'], target['scores'], 0.9)
    boxes = target['boxes'].detach()[nms]
    labels = target['labels'].detach()[nms]

    min_x = boxes[:, 0].min()
    min_y = boxes[:, 1].min()
    _sum = (boxes[:, 0] - min_x) + (boxes[:, 1] - min_y)
    _difference = (boxes[:, 0] - min_x) - (boxes[:, 1] - min_y)

    top_left = _sum.argmin()
    bottom_right = _sum.argmax()
    bottom_left = _difference.argmin()
    top_right = _difference.argmax()

    fig, (ax0, ax1) = plt.subplots(ncols=2)
    ax0.imshow(img.permute(1, 2, 0))

    startpoints = []
    endpoints = []
    gp = GridPosition(1024, 19)
    # for box in boxes:
    #     ax0.plot(box[0], box[1], 'ro')

    for i, j in enumerate((top_left, top_right, bottom_left, bottom_right)):
        box = boxes[j]
        x, y = get_xy(int(labels[j]))

        startpoints.append(box[:2].tolist())
        _x, _y = gp[x - 1][y - 1]
        endpoints.append([_x - gp.grid_size // 2, _y - gp.grid_size // 2])

        ax0.add_patch(Rectangle((box[0], box[1]),
                                box[2] - box[0],
                                box[3] - box[1],
                                linewidth=1,
                                edgecolor='k' if target['labels'][i] <= 362 else 'w',
                                facecolor='none'))

        # ax0.text(box[0], box[1] + 10, f'{x}{S[y]}', fontsize=10, color='r')
        ax0.text(box[0], box[1] + 10, f'{i} {x}{S[y]}', fontsize=10, color='g')

    print(startpoints)
    print(endpoints)
    size = 1024
    transform = cv2.getPerspectiveTransform(np.array(startpoints, np.float32), np.array(endpoints, np.float32))
    _img = cv2.warpPerspective(np.array(pil_image), transform, (size, size))

    # transform = cv2.getAffineTransform(np.array(startpoints, np.float32), np.array(endpoints, np.float32))
    # size = 1024
    # _img = cv2.warpAffine(np.array(pil_image), transform, (size, size))
    ax1.imshow(_img)
    print(gp.x0, gp.y0, gp.x1, gp.y1)
    ax1.add_patch(Rectangle((gp.x0, gp.y0), gp.board_width, gp.board_width, linewidth=1, edgecolor='g', facecolor='none'))

    plt.show()
