import wx


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


class BoardDetector:
    def __init__(self):
        self.bmp = wx.NullBitmap
