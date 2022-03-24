#! /usr/bin/env python3
# coding=utf-8
from re import S
from PIL import Image, ImageDraw, ImageFont
from collections import namedtuple
import random
from sgfmill import sgf, sgf_moves

Point = namedtuple('Point', ['x', 'y'])


class GridPosition:
    def __init__(self, width, size, board_rate=0.8):
        self._board_width = width * board_rate
        x0 = (width - self._board_width) / 2
        y0 = (width - self._board_width) / 2
        self._grid_size = self._board_width / (size - 1)

        self._grid_pos = []
        for y in range(size):
            row_pos = []
            y_pos = int(y0 + self._grid_size * y)
            for x in range(size):
                row_pos.append(Point(int(x0 + self._grid_size * x), y_pos))
            self._grid_pos.append(row_pos)

        self._grid_pos = self._grid_pos[::-1]

        self._star_coords = self.__get_star_point_coords(size)

    def __get_star_point_coords(self, size):
        if size < 7:
            return []
        elif size <= 11:
            star_point_pos = 3
        else:
            star_point_pos = 4

        return [star_point_pos - 1, size - star_point_pos] + (
            [int(size / 2)] if size % 2 == 1 and size > 7 else []
        )

    @property
    def star_coords(self):
        return self._star_coords

    @property
    def grid_size(self):
        return self._grid_size

    @property
    def half_grid_size(self):
        return self._grid_size / 2

    @property
    def board_width(self):
        return self._board_width

    # left
    @property
    def x0(self):
        return self._grid_pos[0][0].x

    # top
    @property
    def y0(self):
        return self._grid_pos[-1][-1].y

    # right
    @property
    def x1(self):
        return self._grid_pos[-1][-1].x

    # bottom
    @property
    def y1(self):
        return self._grid_pos[0][0].y

    def __getitem__(self, index):
        return self._grid_pos[index]


class BaseGenerator:
    DEFAULT_WIDTH = 1024
    BOARD_RATE = 0.8

    def __init__(self, theme):
        self.theme = theme
        self.font = ImageFont.truetype(self.theme['font'], int(self.DEFAULT_WIDTH * 0.02))


class BoardImageGenerator(BaseGenerator):
    def __init__(self, theme, with_coordinates=True):
        super(BoardImageGenerator, self).__init__(theme)
        self.with_coordinates = with_coordinates
        self._board_image = Image.open(self.theme['board'])

    def get_board_image(self, size=19):
        w, h = self._board_image.size
        if self.theme['board_resize']:
            board_image = self._board_image.resize((self.DEFAULT_WIDTH, self.DEFAULT_WIDTH)).convert('RGB')
        else:
            board_image = Image.new('RGB', (self.DEFAULT_WIDTH, self.DEFAULT_WIDTH))
            for x in range(0, self.DEFAULT_WIDTH, w):
                for y in range(0, self.DEFAULT_WIDTH, h):
                    board_image.paste(self._board_image, (x, y))

        draw = ImageDraw.ImageDraw(board_image)
        grid_pos = GridPosition(self.DEFAULT_WIDTH, size, self.BOARD_RATE)

        # draw lines
        for i in range(size):
            draw.line((grid_pos[i][0].x, grid_pos[i][0].y, grid_pos[i][-1].x, grid_pos[i][0].y), self.theme['line_color'])
            draw.line((grid_pos[0][i].x, grid_pos[0][i].y, grid_pos[-1][i].x, grid_pos[-1][i].y), self.theme['line_color'])

        # draw stars
        start_size = self.DEFAULT_WIDTH * 0.005
        for x in grid_pos.star_coords:
            for y in grid_pos.star_coords:
                _x, _y = grid_pos[y][x]
                draw.ellipse((_x - start_size,
                              _y - start_size,
                              _x + start_size,
                              _y + start_size),
                             fill=self.theme['line_color'])

        if self.with_coordinates:
            # draw coordinates
            grid_size = grid_pos.grid_size
            fw, fh = self.font.getsize('A')
            for i in range(size):
                draw.text((grid_pos[0][i].x, grid_pos[0][i].y + grid_size),
                          chr(ord('A') + i),
                          fill=self.theme['line_color'],
                          font=self.font,
                          anchor='mm')
                draw.text((grid_pos[-1][i].x, grid_pos[-1][i].y - grid_size),
                          chr(ord('A') + i),
                          fill=self.theme['line_color'],
                          font=self.font,
                          anchor='mm')
                draw.text((grid_pos[i][0].x - grid_size - fw, grid_pos[i][0].y),
                          str(i + 1),
                          fill=self.theme['line_color'],
                          font=self.font,
                          anchor='mm')
                draw.text((grid_pos[i][-1].x + grid_size, grid_pos[i][-1].y),
                          str(i + 1),
                          fill=self.theme['line_color'],
                          font=self.font,
                          anchor='mm')

        return board_image


class StoneImageGenerator(BaseGenerator):
    def __init__(self, theme):
        super(StoneImageGenerator, self).__init__(theme)
        self._org_stone_images = {'b': [Image.open(b) for b in self.theme['black']],
                                  'w': [Image.open(w) for w in self.theme['white']]}
        self._stone_images = {}

    def get_stone_image(self, color, size):
        if size not in self._stone_images:
            grid_pos = GridPosition(self.DEFAULT_WIDTH, size, self.BOARD_RATE)
            stone_size = int(grid_pos.grid_size * 0.9 * self.theme['scaling_ratio'])
            self._stone_images[size] = {'b': [img.resize((stone_size, stone_size)) for img in self._org_stone_images['b']],
                                        'w': [img.resize((stone_size, stone_size)) for img in self._org_stone_images['w']]}
        imgs = self._stone_images[size][color]
        return imgs[random.randint(0, len(imgs) - 1)]


class GameImageGenerator(BoardImageGenerator, StoneImageGenerator):
    def __init__(self, *args, **kwargs):
        super(GameImageGenerator, self).__init__(*args, **kwargs)

    def _get_sgf_info(self, sgf_path, end=None):
        try:
            sgf_game = sgf.Sgf_game.from_bytes(open(sgf_path, 'rb').read())
        except ValueError:
            raise Exception("bad sgf file")

        try:
            board, plays = sgf_moves.get_setup_and_moves(sgf_game)
        except ValueError as e:
            raise Exception(str(e))

        for i, (colour, move) in enumerate(plays, start=1):
            if move is None:
                continue

            row, col = move
            try:
                board.play(row, col, colour)
            except ValueError:
                raise Exception("illegal move in sgf file")

            if i == end:
                break

        return board, plays[:end]

    def get_game_image(self, sgf_path, img_size=1024, start_number=None, start=None, end=None, board_rate=0.8):
        if img_size != self.DEFAULT_WIDTH:
            self.DEFAULT_WIDTH = img_size
            self.font = ImageFont.truetype(self.theme['font'], int(self.DEFAULT_WIDTH * 0.02))

        self.BOARD_RATE = board_rate

        board, plays = self._get_sgf_info(sgf_path, end)
        if end is None:
            end = len(plays)

        grid_pos = GridPosition(self.DEFAULT_WIDTH, board.side, self.BOARD_RATE)

        board_image = self.get_board_image(board.side).copy()

        stone_offset = int(self.get_stone_image('b', board.side).size[0] // 2 // self.theme['scaling_ratio'])
        stone_offset += int(stone_offset * self.theme['adjust_ratio'])

        draw = ImageDraw.ImageDraw(board_image)
        if start_number is None:
            start_number = start

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
                    board_image.paste(stone_image,
                                      (grid_pos[row][col].x - stone_offset + x_offset,
                                       grid_pos[row][col].y - stone_offset + y_offset),
                                      stone_image)

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

        return board_image
