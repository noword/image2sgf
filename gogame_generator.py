from sgf2img import GameImageGenerator, GridPosition
from sgfmill.boards import Board
import numpy as np
from PIL import ImageFont, ImageDraw
import random
from misc import BoxPostion


class BoardGenerator(GameImageGenerator):
    def get_game_image(self, sgf_path, img_size=1024, start_number=None, start=None, end=None):
        img = super(BoardGenerator, self).get_game_image(sgf_path, img_size, start_number, start, end)

        box_pos = BoxPostion(self.DEFAULT_WIDTH, 19)
        labels = list(range(1, 5))
        boxes = np.array([box_pos[18][0],  # top left
                          box_pos[18][18],  # top right
                          box_pos[0][0],  # bottom left
                          box_pos[0][18]  # bottom right
                          ])
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
    def get_game_image(self, sgf_path, img_size=1024, start_number=None, start=None, end=None):
        if img_size != self.DEFAULT_WIDTH:
            self.DEFAULT_WIDTH = img_size
            self.font = ImageFont.truetype(self.theme['font'], int(self.DEFAULT_WIDTH * 0.02))

        board, plays = self._get_sgf_info(sgf_path, end)
        assert board.side == 19

        if end is None:
            end = len(plays)

        grid_pos = GridPosition(self.DEFAULT_WIDTH, board.side)

        board_image = self.get_board_image(board.side).copy()

        stone_offset = int(self.get_stone_image('b', board.side).size[0] // 2 // self.theme['scaling_ratio'])
        stone_offset += int(stone_offset * self.theme['adjust_ratio'])

        draw = ImageDraw.ImageDraw(board_image)
        if start_number is None:
            start_number = start

        stone_mask, box = get_stone_mask_box(self.get_stone_image('b', board.side))
        # masks = []
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
            else:
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

                    labels.append(row * 19 + col + (0 if color == 'b' else 361) + 1)
                    # mask = np.zeros(board_image.size, dtype=np.uint8)
                    # mask[y0:y0 + stone_mask.shape[1], x0:x0 + stone_mask.shape[0]] = stone_mask
                    # masks.append(mask)
                    boxes.append(box + [x0, y0, x0, y0])

                if start and end >= start:
                    # draw number
                    draw.text((grid_pos[row][col].x + x_offset, grid_pos[row][col].y + y_offset),
                              str(end),
                              fill=num_color[color],
                              font=self.font,
                              anchor='mm')
            end -= 1

        for i, (x, y) in enumerate(((0, 0), (18, 0), (0, 18), (18, 18))):
            if not board.get(x, y):
                labels.append(723 + i)
                x0, y0 = grid_pos[x][y]
                x0 -= grid_pos.grid_size // 2
                y0 -= grid_pos.grid_size // 2
                boxes.append(box + [x0, y0, x0, y0])

        if start:
            for counts in filter(lambda x: len(x) > 1, coor.values()):
                print(' = '.join([str(c) for c in counts]))

        # return board_image, labels, boxes, masks
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
    def __init__(self, max_hands=361, *args, **kwargs):
        super(RandomGenerator, self).__init__(*args, **kwargs)
        self._board = None
        self._plays = None
        self.max_hands = max_hands
        self.set_random_scaling_ratio()

    def set_random_scaling_ratio(self):
        stone_mask, box = get_stone_mask_box(self.get_stone_image('b', 19))
        gp = GridPosition(self.DEFAULT_WIDTH, 19)
        self.theme._theme['scaling_ratio'] *= random.uniform(1, (gp.grid_size - 3) / (box[2] - box[0]))

    def _get_sgf_info(self, sgf_path, end=None):
        if self._board is None:
            self._board = Board(19)
            self._plays = []

            num = random.randint(1, self.max_hands)
            stone_num = 19 * 19 * 2
            samples = list(range(stone_num)) * (num // stone_num + 1)
            for i in random.sample(samples, num):
                x, y, color = STONE_INDEX[i]
                try:
                    self._board.play(x, y, color)
                except ValueError:
                    continue
                self._plays.append((color, (x, y)))

        return self._board, self._plays[:end]


class RandomGogameGenerator(RandomGenerator, GogameGenerator):
    pass


class RandomBoardGenerator(RandomGenerator, BoardGenerator):
    pass
