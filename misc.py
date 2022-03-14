from sgf2img import GridPosition
import numpy as np


def get_xy(idx):
    if idx < 723:
        idx -= 1
        if idx >= 361:
            idx -= 361
        return idx // 19 + 1, idx % 19 + 1
    else:
        idx -= 723
        return ((1, 1), (19, 1), (1, 19), (19, 19))[idx]


S = ' abcdefghijklmnopqrs'


def get_xy_string(idx):
    x, y = get_xy(idx)
    return f'{S[x]}{S[y]}'


def get_4points_from_box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]


class BoxPostion(GridPosition):
    def __init__(self, width, size, board_rate=0.8):
        super(BoxPostion, self).__init__(width, size, board_rate)
        self._boxes = []
        for row in self._grid_pos:
            self._boxes.append([[max(x - self.half_grid_size, 0),
                                 max(y - self.half_grid_size, 0),
                                 min(x + self.half_grid_size, width),
                                 min(y + self.half_grid_size, width)] for x, y in row])

    @property
    def boxes(self):
        return self._boxes

    def __getitem__(self, index):
        return self._boxes[index]


class NpBoxPostion(BoxPostion):
    def __init__(self, *args, **kwargs):
        super(NpBoxPostion, self).__init__(*args, **kwargs)
        self._boxes = np.array(self._boxes)
