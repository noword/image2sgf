from .gogame_generator import GogameGenerator, RandomGogameGenerator, RandomBoardGenerator
from .misc import get_xy, S
from .sgf2img import GetAllThemes
import random
import numpy as np
import torch
import pathlib
import gc
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
from PIL import ImageDraw


class GogameDataset(torch.utils.data.Dataset):
    def __init__(self, initvar='./sgf/', transforms=None):
        self.transforms = transforms
        self.sgfs = []
        self.themes = list(GetAllThemes().values())
        self.initseed()
        self._initsgfs(initvar)
        assert len(self.sgfs) > 0

    def initseed(self):
        random.seed(datetime.now())
        self.seed = random.randint(0, 0xffffffff)

    def _initsgfs(self, var):
        for path in pathlib.Path(var).glob('*.sgf'):
            self.sgfs.append(str(path))

    @property
    def _generator(self):
        theme = self.themes[random.randint(0, len(self.themes) - 1)]
        return GogameGenerator(theme=theme, with_coordinates=bool(random.getrandbits(1)))

    def __len__(self):
        return len(self.sgfs)

    def __getitem__(self, idx):
        random.seed(self.seed + idx)
        torch.manual_seed(self.seed + idx)

        gig = self._generator

        board, setups, plays = gig._get_sgf_info(self.sgfs[idx])
        num_plays = len(plays)

        end = random.randint(max(1, int(num_plays * 0.8)), len(plays))
        start = None if bool(random.getrandbits(1)) else random.randint(1, end)
        start_number = None if bool(random.getrandbits(1)) else 1
        board_rate = random.uniform(0.8, 0.94)
        img, labels, boxes = gig.get_game_image(self.sgfs[idx], 1024, start_number, start, end, board_rate)
        boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        target = {'labels': torch.as_tensor(labels, dtype=torch.int64),
                  'boxes': boxes,
                  'image_id': torch.tensor([idx]),
                  'iscrowd': torch.zeros([len(labels)], dtype=torch.int64),
                  'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                  }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            img[img > 1.] = 1.

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

        def _show(event):
            nonlocal count

            if event and event.key == 'escape':
                plt.close('all')
                return

            ax.clear()

            img, target = self[count]
            ax.imshow((img.permute(1, 2, 0) * 255).to(torch.uint8))

            for i, box in enumerate(target['boxes']):
                if target['labels'][i] <= 361:
                    c = 'k'
                elif target['labels'][i] <= 722:
                    c = 'w'
                else:
                    c = 'r'

                ax.add_patch(Rectangle((box[0], box[1]),
                                       box[2] - box[0],
                                       box[3] - box[1],
                                       linewidth=1,
                                       edgecolor=c,
                                       facecolor='none'))
                x, y = get_xy(int(target['labels'][i]))
                ax.text(box[0], box[1] + 10, f'{x}{S[y]}', fontsize=10, color='g' if target['labels'][i] <= 723 else 'r')

            if 'keypoints' in target:
                for kp in target['keypoints']:
                    for x, y, _ in kp:
                        ax.plot(x, y, 'ro', markersize=2)

            fig.canvas.draw()

            self.save(count)
            # print(target)
            count += 1
            if count >= len(self):
                count = 0

        fig.canvas.mpl_connect('key_press_event', _show)
        _show(None)
        plt.show()


class RandomGogameDataset(GogameDataset):
    def __init__(self, initvar=1000, hands_num=(1, 361), transforms=None):
        self.hands_num = hands_num
        super(RandomGogameDataset, self).__init__(initvar=initvar, transforms=transforms)

    def _initsgfs(self, var):
        self.sgfs = [str(i) for i in range(var)]

    @property
    def _generator(self):
        theme = self.themes[random.randint(0, len(self.themes) - 1)]
        return RandomGogameGenerator(hands_num=self.hands_num, theme=theme, with_coordinates=bool(random.getrandbits(1)))


class RandomBoardDataset(RandomGogameDataset):
    def __init__(self, *args, **kwargs):
        super(RandomBoardDataset, self).__init__(*args, **kwargs)

    @property
    def _generator(self):
        theme = self.themes[random.randint(0, len(self.themes) - 1)]
        return RandomBoardGenerator(hands_num=self.hands_num, theme=theme, with_coordinates=bool(random.getrandbits(1)))


if __name__ == '__main__':
    from model import get_transform
    d = RandomBoardDataset(transforms=get_transform(True))
    d.show()
