from img2sgf import get_models, get_board_image, classifier_board, classifer_part_board, NpBoxPostion, DEFAULT_IMAGE_SIZE, get_sgf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import argparse
import time


def demo(pil_img, save_images=False):
    if isinstance(pil_img, str):
        pil_img = Image.open(pil_img).convert('RGB')
    print('1st perspective')
    img0, boxes0, scores0 = get_board_image(board_model, pil_img)
    print('2nd perspective')
    img1, boxes1, scores1 = get_board_image(board_model, img0)
    img = img0 if sum(scores0) > sum(scores1) else img1
    board = classifier_board(stone_model, img, save_images)

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


def part_demo(pil_img, save_images=False):
    if isinstance(pil_img, str):
        pil_img = Image.open(pil_img).convert('RGB')
    boxes, labels, scores, results = classifer_part_board(part_board_model, stone_model, pil_img, save_images)
    p = plt.imshow(pil_img)
    for i, box in enumerate(boxes):
        p.add_patch(Rectangle((box[0], box[1]),
                              box[2] - box[0],
                              box[3] - box[1],
                              linewidth=1,
                              edgecolor='g',
                              facecolor='none'))
    plt.show()


def img2sgf(img, sgf_name, save_images=False):
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    _img, _, scores = get_board_image(board_model, img)
    if min(scores) < 0.7:
        _img0, boxes0, scores0 = get_board_image(board_model, _img)
        if sum(scores0) > sum(scores):
            _img, boxes, scores = _img0, boxes0, scores0

    board = classifier_board(stone_model, _img, save_images)
    sgf = get_sgf(board)
    open(sgf_name, 'wb').write(sgf.serialise())


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('image_name', action='store', nargs='?', help='input image file name')
    parser.add_argument('--sgf_name', action='store', help='output sgf file name')
    parser.add_argument('--capture', action='store_true', default=False, help='capture the screenshot')
    parser.add_argument('--save_images', action='store_true', default=False, help='save grid images')
    args = parser.parse_args()

    board_model, part_board_model, stone_model = get_models()

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
