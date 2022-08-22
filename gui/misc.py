from PIL import Image
import wx
import numpy
import cv2


def img_wx_to_pil(img):
    buf = bytes(img.GetData())
    w, h = img.GetSize()
    img = Image.new('RGB', (w, h))
    img.frombytes(buf)
    return img


def img_pil_to_wx(img):
    w, h = img.size
    return wx.ImageFromBuffer(w, h, img.tobytes())


def img_wx_to_cv2(img):
    buf = bytes(img.GetData())
    w, h = img.GetSize()
    img = numpy.ndarray(shape=(h, w, 3), dtype=numpy.uint8, buffer=buf)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def img_cv2_to_wx(img):
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return wx.ImageFromBuffer(w, h, img)


def img_pil_to_cv2(img):
    return numpy.array(img)[:, :, ::-1]


def img_cv2_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def to_cv2(img):
    if isinstance(img, wx.Image):
        return img_wx_to_cv2(img)
    elif isinstance(img, Image.Image):
        return img_pil_to_cv2(img)
    elif isinstance(img, numpy.ndarray):
        return img
    else:
        raise TypeError


def to_pil(img):
    if isinstance(img, wx.Image):
        return img_wx_to_pil(img)
    elif isinstance(img, numpy.ndarray):
        return img_cv2_to_pil(img)
    elif isinstance(img, Image.Image):
        return img
    else:
        raise TypeError


def to_wx(img):
    if isinstance(img, Image.Image):
        return img_pil_to_wx(img)
    elif isinstance(img, numpy.ndarray):
        return img_cv2_to_wx(img)
    elif isinstance(img, wx.Image):
        return img
    else:
        raise TypeError
