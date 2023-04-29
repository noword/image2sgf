import wx
from gui.main import *


class App(wx.App):
    def InitLocale(self):
        import sys
        if sys.platform.startswith('win') and sys.version_info > (3, 8):
            import locale
            locale.setlocale(locale.LC_ALL, "C")


def run():
    app = App(True, 'img2sgf.log')
    frame = MainFrame(None, title='img2sgf v0.07')
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    run()
