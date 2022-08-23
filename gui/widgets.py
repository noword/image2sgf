import wx
from .misc import to_wx


class DragShape:
    def __init__(self, bmp):
        self.bmp = bmp
        self.pos = (0, 0)
        self.shown = True
        self.text = None
        self.fullscreen = False

    def HitTest(self, pt):
        rect = self.GetRect()
        return rect.Contains(pt)

    def GetRect(self):
        return wx.Rect(self.pos[0], self.pos[1],
                       self.bmp.GetWidth(), self.bmp.GetHeight())

    def Draw(self, dc, op=wx.COPY):
        if self.bmp.IsOk():
            memDC = wx.MemoryDC()
            memDC.SelectObject(self.bmp)

            dc.Blit(self.pos[0], self.pos[1],
                    self.bmp.GetWidth(), self.bmp.GetHeight(),
                    memDC, 0, 0, op, True)

            return True
        else:
            return False


class ImagePanel(wx.Panel):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.bmp = wx.StaticBitmap(self)
        self.img = None
        self.Bind(wx.EVT_SIZE, self.OnSize)

    def SetImage(self, img):
        if img:
            self.img = to_wx(img)
            self._SetBmp()

    def IsEmpty(self):
        return self.img is None

    def _SetBmp(self):
        def rescale(img, w, h):
            ow, oh = img.GetSize()
            if w / h > ow / oh:
                new_w = int(h / oh * ow)
                new_h = h
            else:
                new_w = w
                new_h = int(w / ow * oh)
            return img.Copy().Rescale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)

        img = self.img
        if img is None:
            bmp = wx.NullBitmap
        else:
            w, h = self.GetSize()
            img = rescale(img, w, h)
            bmp = wx.Bitmap(img)
            x = max(0, (w - img.Width) // 2)
            y = max(0, (h - img.Height) // 2)
            self.bmp.SetPosition(wx.Point(x, y))
        self.bmp.SetBitmap(bmp)

    def OnSize(self, event):
        self._SetBmp()
        event.Skip()


class BoardDetector:
    def __init__(self):
        self.bmp = wx.NullBitmap


if __name__ == '__main__':
    # app = wx.App()
    # frame = Frame()
    # frame.Show()
    # app.MainLoop()
    app = wx.App()
    window = wx.Frame(None, title="wxPython Frame", size=(300, 200))
    panel = ImagePanel(parent=window, img=wx.Image('../IMG_20220817_130349.jpg'))

    window.Show(True)
    app.MainLoop()
