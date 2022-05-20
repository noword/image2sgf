# Dependencies
[python](https://www.python.org/) >= 3.6

[PyTorch](https://pytorch.org/)

[torchvision](https://pytorch.org/vision/stable/index.html)

[cv2](https://github.com/opencv/opencv-python)

[numpy](https://numpy.org/)

[matplotlib](https://matplotlib.org/)

[pillow](https://pillow.readthedocs.io/en/stable/)

[pyautogui](https://pypi.org/project/PyAutoGUI/)


# Usage:
Download pth files from https://github.com/noword/image2sgf/releases/tag/v0.1

```
usage: img2sgf.py [-h] [--sgf_name SGF_NAME] [--capture] [--save_images] [image_name]

positional arguments:
  image_name           input image file name

optional arguments:
  -h, --help           show this help message and exit
  --sgf_name SGF_NAME  output sgf file name
  --capture            capture the screenshot
  --save_images        save grid images
  ```

# Examples:

`img2sgf.py examples\goban.png`
![](example/1.jpg)



