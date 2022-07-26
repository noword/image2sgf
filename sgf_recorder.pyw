import wx
from img2sgf import get_models, get_board_image, classifier_board, NpBoxPostion, DEFAULT_IMAGE_SIZE, get_sgf
from img2sgf.sgf2img import GameImageGenerator, Theme, GetAllThemes
import recorder_imgs

_ = wx.GetTranslation


class MainFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MainFrame, self).__init__(parent,
                                        title=title,
                                        size=(840, 480))
        self.SetIcon(recorder_imgs.GO.GetIcon())
        self.Center()

        self.toolbar = self.CreateToolBar(wx.TB_FLAT)
        self.toolbar.SetToolBitmapSize((32, 32))

        source_cbox = wx.ComboBox(self.toolbar, size=(150, -1), style=wx.CB_DROPDOWN)
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
