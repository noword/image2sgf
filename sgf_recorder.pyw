import wx
from img2sgf import get_models, get_board_image, classifier_board, NpBoxPostion, DEFAULT_IMAGE_SIZE, get_sgf
from img2sgf.sgf2img import GameImageGenerator, Theme, GetAllThemes
import recorder_imgs
import cv2

_ = wx.GetTranslation


def get_camera_num():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            index += 1
            cap.release()
    return index


class MainFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MainFrame, self).__init__(parent,
                                        title=title,
                                        size=(840, 480))
        self.SetIcon(recorder_imgs.GO.GetIcon())
        self.Center()

        self.toolbar = self.CreateToolBar(wx.TB_FLAT)
        self.toolbar.SetToolBitmapSize((32, 32))

        self.toolbar.AddControl(wx.StaticText(self.toolbar, label=_('Video Stream Source')))
        source_cbox = wx.ComboBox(self.toolbar,
                                  value=_('Screenshot'),
                                  size=(150, -1),
                                  choices=[_('Screenshot')] + [_('Camera ') + str(i) for i in range(get_camera_num())],
                                  style=wx.CB_DROPDOWN)
        self.Bind(wx.EVT_COMBOBOX, self.OnVideoSourceChanged, source_cbox)
        self.toolbar.AddControl(source_cbox)

        self.toolbar.AddTool(10,
                             _('Detect'),
                             recorder_imgs.DETECT.GetBitmap(),
                             _('Detect the board'))

        self.toolbar.AddSeparator()

        self.toolbar.AddTool(20,
                             _('Record'),
                             recorder_imgs.RECORD.GetBitmap(),
                             _('Start recording'))

        self.toolbar.AddTool(30,
                             _('Pause'),
                             recorder_imgs.PAUSE.GetBitmap(),
                             _('Pause'))
        self.toolbar.AddTool(40,
                             _('Stop'),
                             recorder_imgs.STOP.GetBitmap(),
                             _('Stop'))
        self.toolbar.AddSeparator()

        self.toolbar.AddTool(50,
                             _('Left'),
                             recorder_imgs.LEFT.GetBitmap(),
                             _('Rotate left'))
        # self.Bind(wx.EVT_TOOL, self.OnRotateClick, id=40)

        self.toolbar.AddTool(60,
                             _('Right'),
                             recorder_imgs.RIGHT.GetBitmap(),
                             _('Rotate right'))

        self.toolbar.AddSeparator()

        self.toolbar.AddTool(70,
                             _('Option'),
                             recorder_imgs.OPTIONS.GetBitmap(),
                             _('Option'),
                             )
        # self.Bind(wx.EVT_TOOL, self.OnOptionClick, id=60)

        self.toolbar.AddTool(80,
                             _('Home'),
                             recorder_imgs.HOME.GetBitmap(),
                             _('Home page'))
        # self.Bind(wx.EVT_TOOL, self.OnHomeClick, id=70)
        self.toolbar.Realize()

        self.client = wx.Panel(self)
        self.images = [None] * 4
        sizer = wx.GridSizer(2, 2, 1, 1)
        self.bitmaps = [wx.StaticBitmap(self.client) for i in range(4)]
        sizer.AddMany(self.bitmaps)
        self.client.SetSizer(sizer)

        self.status = self.CreateStatusBar(1)

    def OnVideoSourceChanged(self, event):
        print(event.GetSelection())


class App(wx.App):
    def InitLocale(self):
        import sys
        if sys.platform.startswith('win') and sys.version_info > (3, 8):
            import locale
            locale.setlocale(locale.LC_ALL, "C")


def run():
    app = App(True, 'sgf_recorder.log')
    frame = MainFrame(None, title='sgf_recorder v0.01')
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    run()
