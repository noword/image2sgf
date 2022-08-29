from .sgf2img import GameImageGenerator, GridPosition
from .misc import BoxPostion
from sgfmill.boards import Board
import numpy as np
from PIL import ImageFont, ImageDraw
import random


class BoardGenerator(GameImageGenerator):
    def get_game_image(self, sgf_path, img_size=1024, start_number=None, start=None, end=None, part_rect=None):
        img = super(BoardGenerator, self).get_game_image(sgf_path, img_size, start_number, start, end, part_rect)

        box_pos = BoxPostion(self.DEFAULT_WIDTH, 19)
        labels = list(range(1, 5))
        boxes = np.array([box_pos[18][0],  # top left
                          box_pos[18][18],  # top right
                          box_pos[0][0],  # bottom left
                          box_pos[0][18]  # bottom right
                          ])
        return img, labels, boxes


class PartBoardGenerator(GameImageGenerator):
    def get_game_image(self, sgf_path, img_size=1024, start_number=None, start=None, end=None, part_rect=None):
        img = super(PartBoardGenerator, self).get_game_image(sgf_path, img_size, start_number, start, end, part_rect)
        if part_rect is None:
            part_rect = [1, 1, 19, 19]

        print(part_rect)
        part_rect = [x if x == 0 else x - 1 for x in part_rect]
        box_pos = BoxPostion(self.DEFAULT_WIDTH, 19)
        x0, y0, x1, y1 = part_rect
        labels = [x0 + y0 * 19 + 1, x1 + y0 * 19 + 1, x0 + y1 * 19 + 1, x1 + y1 * 19 + 1]
        boxes = [box_pos[y0][x0], box_pos[y0][x1], box_pos[y1][x0], box_pos[y1][x1]]
        x0_offset, y0_offset, _, _ = box_pos[y1][x0]
        if y1 >= 18:
            y0_offset = 0
        if x0 <= 0:
            x0_offset = 0
        boxes = [[_x0 - x0_offset, _y0 - y0_offset, _x1 - x0_offset, _y1 - y0_offset] for _x0, _y0, _x1, _y1 in boxes]
        print(x0_offset, y0_offset, part_rect)

        return img, labels, boxes


def get_stone_mask_box(img):
    alpha = img.getchannel('A')
    alpha = alpha.point(lambda x: 1 if x == 255 else 0)
    alpha = np.array(alpha).astype(np.uint8)
    pos = np.where(alpha)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    return alpha, np.array([xmin, ymin, xmax, ymax])


class GogameGenerator(GameImageGenerator):
    def _filter(self, part_rect, row, col):
        if part_rect is None:
            return True
        left, top, right, bottom = part_rect
        return top < row < bottom and left < col < right

    def get_game_image(self, sgf_path, img_size=1024, start_number=None, start=None, end=None, board_rate=0.8, part_rect=None):
        if img_size != self.DEFAULT_WIDTH:
            self.DEFAULT_WIDTH = img_size
            self.font = ImageFont.truetype(self.theme['font'], int(self.DEFAULT_WIDTH * 0.02))

        self.BOARD_RATE = board_rate

        board, _, plays = self._get_sgf_info(sgf_path, end)
        assert board.side == 19

        if end is None:
            end = len(plays)

        grid_pos = GridPosition(self.DEFAULT_WIDTH, board.side, board_rate)

        board_image = self.get_board_image(board.side).copy()

        stone_offset = int(self.get_stone_image('b', board.side).size[0] // 2 // self.theme['scaling_ratio'])
        stone_offset += int(stone_offset * self.theme['adjust_ratio'])

        draw = ImageDraw.ImageDraw(board_image)
        if start_number is None:
            start_number = start

        stone_mask, box = get_stone_mask_box(self.get_stone_image('b', board.side))
        # masks = []
        if part_rect:
            part_rect = [x - 1 for x in part_rect]

        labels = []
        boxes = []

        coor = {}
        num_color = {'b': 'white',
                     'w': self.theme['line_color'],
                     None: self.theme['line_color']}
        for colour, move in plays[::-1]:
            if move is None:
                continue

            row, col = move
            if move in coor:
                coor[move].append(end)
            elif self._filter(part_rect, row, col):
                coor[move] = [end]
                color = board.get(row, col)
                x_offset = y_offset = 0

                # draw stone
                if color:
                    stone_image = self.get_stone_image(color, board.side)
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)
                    x0 = grid_pos[row][col].x - stone_offset + x_offset
                    y0 = grid_pos[row][col].y - stone_offset + y_offset
                    board_image.paste(stone_image,
                                      (x0, y0),
                                      stone_image)

                    labels.append(row * 19 + col + 1)
                    boxes.append(box + [x0, y0, x0, y0])

                if start and end >= start:
                    # draw number
                    draw.text((grid_pos[row][col].x + x_offset, grid_pos[row][col].y + y_offset),
                              str(end),
                              fill=num_color[color],
                              font=self.font,
                              anchor='mm')
            end -= 1

        if start:
            for counts in filter(lambda x: len(x) > 1, coor.values()):
                print(' = '.join([str(c) for c in counts]))

        if part_rect:
            if len(boxes) == 0:
                col = part_rect[0] + (part_rect[2] - part_rect[0]) // 2
                row = part_rect[1] + (part_rect[3] - part_rect[1]) // 2

                stone_image = self.get_stone_image('b', board.side)
                x_offset = random.randint(-1, 1)
                y_offset = random.randint(-1, 1)
                x0 = grid_pos[row][col].x - stone_offset + x_offset
                y0 = grid_pos[row][col].y - stone_offset + y_offset
                board_image.paste(stone_image,
                                  (x0, y0),
                                  stone_image)

                labels.append(row * 19 + col + 1)
                boxes.append(box + [x0, y0, x0, y0])

            rect = []
            part_rect[1], part_rect[3] = part_rect[3], part_rect[1]
            for i in part_rect:
                if i <= 1:
                    v = 0
                elif i >= board.side - 1:
                    v = img_size
                else:
                    v = grid_pos[i][i].x + grid_pos.half_grid_size
                rect.append(v)

            rect[1] = img_size - rect[1]
            rect[3] = img_size - rect[3]

            board_image = board_image.crop(rect)
            boxes = np.array(boxes)
            boxes[:, ::2] -= int(rect[0])
            boxes[:, 1::2] -= int(rect[1])

        return board_image, labels, boxes


def GenStoneIndex():
    stones = {}
    for i in range(19 * 19):
        x = i % 19
        y = i // 19
        stones[i] = (x, y, 'b')
        stones[i + 19 * 19] = (x, y, 'w')
    return stones


STONE_INDEX = GenStoneIndex()


class RandomGenerator:
    def __init__(self, hands_num=(1, 361), *args, **kwargs):
        super(RandomGenerator, self).__init__(*args, **kwargs)
        self._board = None
        self._plays = None
        self.hands_num = hands_num
        self.set_random_scaling_ratio()

    def set_random_scaling_ratio(self):
        stone_mask, box = get_stone_mask_box(self.get_stone_image('b', 19))
        gp = GridPosition(self.DEFAULT_WIDTH, 19, self.BOARD_RATE)
        self.theme._theme['scaling_ratio'] *= random.uniform(1, (gp.grid_size - 3) / (box[2] - box[0]))

    def _get_sgf_info(self, sgf_path, end=None):
        if self._board is None:
            self._board = Board(19)
            self._plays = []

            num = random.randint(*self.hands_num)
            stone_num = 19 * 19 * 2
            samples = list(range(stone_num)) * (num // stone_num + 1)
            for i in random.sample(samples, num):
                x, y, color = STONE_INDEX[i]
                try:
                    self._board.play(x, y, color)
                except ValueError:
                    continue
                self._plays.append((color, (x, y)))

        return self._board, [[], [], []], self._plays[:end]


class RandomGogameGenerator(RandomGenerator, GogameGenerator):
    pass


class RandomBoardGenerator(RandomGenerator, BoardGenerator):
    pass


class RandomPartBoardGenerator(RandomGenerator, PartBoardGenerator):
    pass
