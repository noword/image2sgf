import torch
from .sgf2img import GetAllThemes
from .gogame_generator import RandomBoardGenerator
from PIL import Image, ImageDraw
from .misc import NpBoxPostion
import os
import random
import pathlib

COLOR_INDEX = {None: 0, 'b': 1, 'w': 2}
GEO = '△◯◻╳'


class StoneDataset(torch.utils.data.Dataset):
    # label bits:
    # [offset]    [context]
    # 0         with or without sign (number, letter or geometry)
    # 1-2       00: empty
    #           01: black
    #           10: white

    def __init__(self, theme_path=None, data_path=None, transforms=None):
        self.transforms = transforms
        self.items = []

        if theme_path:
            self._load_theme(theme_path)

        if data_path:
            self._load_data(data_path)

    def _load_theme(self, theme_path):
        for theme in GetAllThemes(theme_path).values():
            gig = RandomBoardGenerator(theme=theme, hands_num=(180, 361))
            sign_color = {'b': 'white',
                          'w': gig.theme['line_color'],
                          None: gig.theme['line_color']}

            board, _, plays = gig._get_sgf_info(None)
            box_pos = NpBoxPostion(gig.DEFAULT_WIDTH, board.side, gig.BOARD_RATE)
            offset = int(box_pos.grid_size * 0.2)

            img = gig.get_game_image(None)[0]
            draw = ImageDraw.ImageDraw(img)
            for x in range(board.side):
                for y in range(board.side):
                    color = board.get(x, y)
                    label = COLOR_INDEX[color] << 1
                    if bool(random.getrandbits(1)):
                        label |= 1
                        p = random.random()
                        if p > 0.4:
                            s = str(random.randint(1, 400))
                        elif p > 0.2:
                            s = chr(random.randint(0x41, 0x5a) + (0 if bool(random.getrandbits(1)) else 0x20))
                        else:
                            s = GEO[random.randint(0, 3)]
                        draw.text(box_pos._grid_pos[x][y],
                                  s,
                                  fill=sign_color[color],
                                  font=gig.font,
                                  anchor='mm')
                    x_offset = random.randint(-offset, offset)
                    y_offset = random.randint(-offset, offset)
                    self.items.append((img.crop(box_pos[x][y] + [x_offset, y_offset, x_offset, y_offset]),
                                       label))

    def _load_data(self, data_path):
        for label in range(6):
            for path in pathlib.Path(f'{data_path}/{label}/').glob('*.jpg'):
                self.items.append((Image.open(str(path)), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, label = self.items[idx]
        if self.transforms:
            img, label = self.transforms(img, label)

        return img, label

    def save_all(self):
        import torchvision.transforms as T
        for i in range(6):
            try:
                os.mkdir(str(i))
            except BaseException:
                pass

        for i, (img, label) in enumerate(self):
            if torch.is_tensor(img):
                img = T.ToPILImage()(img)
            img.save(f'{label}/{i}.jpg')


if __name__ == '__main__':
    from model import get_stone_transform
    d = StoneDataset(theme_path='themes',
                     # data_path='stone_data',
                     transforms=get_stone_transform(True))
    d.save_all()
