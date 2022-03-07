from sgf2img import GridPosition


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
    def __init__(self, width, size):
        super(BoxPostion, self).__init__(width, size)
        self._boxes = []
        half_grid_size = self.grid_size / 2
        for row in self._grid_pos:
            self._boxes.append([[x - half_grid_size,
                                 y - half_grid_size,
                                 x + half_grid_size,
                                 y + half_grid_size] for x, y in row])

    @property
    def boxes(self):
        return self._boxes

    def __getitem__(self, index):
        return self._boxes[index]
