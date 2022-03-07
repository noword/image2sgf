from sgf2img import GetAllThemes, GameImageGenerator, GridPosition
from sgfmill.boards import Board
import random
import torch
import pathlib
import torchvision
from random_background import RandomBackground
import transforms as T
import gc
from PIL import ImageDraw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class BoardGenerator(GameImageGenerator):
    def get_game_image(self, sgf_path, img_size=None, start_number=None, start=None, end=None):
        img = super().get_game_image(sgf_path, img_size, start_number, start, end)
        grid_pos = GridPosition(self.DEFAULT_WIDTH, 19)
        size = grid_pos.grid_size // 2
        return img, (grid_pos.x0 - size, grid_pos.y0 - size, grid_pos.x1 + size, grid_pos.y1 + size)


def GenStoneIndex():
    stones = {}
    for i in range(19 * 19):
        x = i % 19
        y = i // 19
        stones[i] = (x, y, 'b')
        stones[i + 19 * 19] = (x, y, 'w')
    return stones


STONE_INDEX = GenStoneIndex()


class RandomGenerator(BoardGenerator):
    def __init__(self, max_hands=361, *args, **kwargs):
        super(RandomGenerator, self).__init__(*args, **kwargs)
        self._board = None
        self._plays = None
        self.max_hands = max_hands

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


class BoardDataset(torch.utils.data.Dataset):
    def _initsgfs(self, var):
        for path in pathlib.Path(var).glob('*.sgf'):
            self.sgfs.append(str(path))

    def __init__(self, initvar='./sgf/', transforms=None):
        self.transforms = transforms
        self.sgfs = []
        self.themes = list(GetAllThemes().values())

        self.seed = random.randint(0, 0xffffffff)
        self._initsgfs(initvar)
        assert len(self.sgfs) > 0

    @property
    def _generator(self):
        theme = self.themes[random.randint(0, len(self.themes) - 1)]
        return BoardGenerator(theme=theme, with_coordinates=bool(random.getrandbits(1)))

    def __len__(self):
        return len(self.sgfs)

    def __getitem__(self, idx):
        random.seed(self.seed + idx)
        torch.manual_seed(self.seed + idx)

        gig = self._generator

        board, plays = gig._get_sgf_info(self.sgfs[idx])
        num_plays = len(plays)

        end = random.randint(max(1, int(num_plays * 0.8)), len(plays))
        start = None if bool(random.getrandbits(1)) else random.randint(1, end)
        start_number = None if bool(random.getrandbits(1)) else 1
        img, box = gig.get_game_image(self.sgfs[idx], None, start_number, start, end)

        w, h = img.size
        x0, y0, x1, y1 = box
        boxes = torch.as_tensor([[0, 0, w, h]], dtype=torch.float32)
        keypoints = torch.as_tensor([[[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]]], dtype=torch.float32)
        target = {'labels': torch.as_tensor([1], dtype=torch.int64),
                  'boxes': boxes,
                  #   'image_id': torch.tensor([idx]),
                  #   'iscrowd': torch.zeros([1], dtype=torch.int64),
                  #   'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                  'keypoints': keypoints
                  }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        gc.collect()
        return img, target

    def save(self, idx, name=None, with_target=False):
        img, target = self[idx]
        if name is None:
            name = f'{idx}.jpg'

        _img = torchvision.transforms.ToPILImage()(img)

        if with_target:
            draw = ImageDraw.ImageDraw(_img)
            for box in target['boxes']:
                draw.rectangle(box.tolist(), outline='green', width=1)
            for keypoint in target['keypoints']:
                for k in keypoint:
                    x, y, v = k
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='red', outline='red')

        _img.save(name)

    def show(self):
        count = 0
        fig, ax = plt.subplots()

        def show(event):
            nonlocal count
            ax.clear()

            img, target = self[count]
            ax.imshow(img.permute(1, 2, 0))

            for box in target['boxes']:
                ax.add_patch(Rectangle((box[0], box[1]),
                                       box[2] - box[0],
                                       box[3] - box[1],
                                       linewidth=1,
                                       edgecolor='g',
                                       facecolor='none'))

            for kp in target['keypoints']:
                for x, y, _ in kp:
                    ax.plot(x, y, 'ro', markersize=2)

            fig.canvas.draw()

            print(target)
            count += 1
            if count >= len(self):
                count = 0

        fig.canvas.mpl_connect('key_press_event', show)
        show(None)
        plt.show()


class RandomBoardDataset(BoardDataset):
    def _initsgfs(self, var):
        self.sgfs = [str(i) for i in range(var)]

    def __init__(self, initvar=1000, max_hands=361, transforms=None):
        self.max_hands = max_hands
        super(RandomBoardDataset, self).__init__(initvar=initvar, transforms=transforms)

    @property
    def _generator(self):
        theme = self.themes[random.randint(0, len(self.themes) - 1)]
        return RandomGenerator(max_hands=self.max_hands, theme=theme, with_coordinates=bool(random.getrandbits(1)))


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(RandomBackground())
    return T.Compose(transforms)


if __name__ == '__main__':
    d = RandomBoardDataset(transforms=get_transform(True))
    d.save(0)
    d.show()
