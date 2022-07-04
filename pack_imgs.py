from wx.tools import img2py
from PIL import Image
import os

SIZE = (32, 32)
TMP_NAME = 'tmp.png'
IMAGES_NAME = 'imgs.py'

if os.path.exists(IMAGES_NAME):
    os.remove(IMAGES_NAME)

for name in ('screenshot', 'open', 'save', 'left', 'right', 'go', 'home', 'options'):
    img = Image.open(f'{name}.png').resize(SIZE)
    img.save(TMP_NAME)
    img2py.img2py(TMP_NAME, IMAGES_NAME, append=os.path.exists(IMAGES_NAME), imgName=name.upper())
