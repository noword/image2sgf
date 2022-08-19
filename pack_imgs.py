from wx.tools import img2py
from PIL import Image
import os

SIZE = (32, 32)
TMP_NAME = 'tmp.png'
IMAGES_NAME = 'imgs.py'


def gen(py_name, images, size=(32, 32), tmp_name='tmp.png'):
    if os.path.exists(py_name):
        os.remove(py_name)

    for name in images:
        img = Image.open(f'images/{name}.png').resize(SIZE)
        img.save(tmp_name)
        img2py.img2py(tmp_name, py_name, append=os.path.exists(py_name), imgName=name.upper())


gen('gui/imgs.py', ('screenshot', 'open', 'save', 'left', 'right', 'go', 'home', 'options'))
gen('recorder_imgs.py', ('go', 'detect', 'save', 'left', 'right', 'record', 'pause', 'stop', 'home', 'options'))
