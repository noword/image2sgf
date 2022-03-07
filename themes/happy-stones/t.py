from PIL import Image

im = Image.open('w.png')
im = im.crop((10, 10, 140, 140))
im.save('glass_white.png')
