import wx
from img2sgf import get_models, get_board_image, classifier_board, NpBoxPostion, DEFAULT_IMAGE_SIZE, get_sgf
from img2sgf.sgf2img import GameImageGenerator, Theme, GetAllThemes
import recorder_imgs
import cv2
import pyautogui
from PIL import Image
import threading
import time
import numpy

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


EVT_NEW_IMAGE = wx.NewIdRef()


class NewImageEvent(wx.PyEvent):
    def __init__(self, index, image):
        super().__init__()
        self.SetEventType(EVT_NEW_IMAGE)
        self.index = index
        self.image = image


class Model:
    def __init__(self, window, board_path='board.pth', stone_path='stone.pth', theme='real-stones'):
        self.window = window
        self.theme = theme
        self.board_model = self.stone_model = None

        def load_model(board_path, stone_path):
            self.window.status.SetStatusText(_('loading models'))
            self.board_model, self.part_board_model, self.stone_model = get_models(board_path, 'part_board.pth', stone_path)
            self.window.toolbar.EnableTool(10, True)
            self.window.status.SetStatusText('')

        threading.Thread(target=load_model, args=(board_path, stone_path)).start()


class MainFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MainFrame, self).__init__(parent,
                                        title=title,
                                        size=(840, 480))
        self.model = Model(self)

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

        _id = 10
        for label, bmp, shorthelp, handler in ((_('Detect'), recorder_imgs.DETECT.GetBitmap(), _('Detect the board'), None),
                                               (None, None, None, None,),
                                               (_('Record'), recorder_imgs.RECORD.GetBitmap(), _('Start recording'), None),
                                               (_('Pause'), recorder_imgs.PAUSE.GetBitmap(), _('Pause'), None),
                                               (_('Stop'), recorder_imgs.STOP.GetBitmap(), _('Stop'), None),
                                               (None, None, None, None,),
                                               (_('Left'), recorder_imgs.LEFT.GetBitmap(), _('Rotate left'), None),
                                               (_('Right'), recorder_imgs.RIGHT.GetBitmap(), _('Rotate right'), None),
                                               (None, None, None, None),
                                               (_('Option'), recorder_imgs.OPTIONS.GetBitmap(), _('Option'), None),
                                               (_('Home'), recorder_imgs.HOME.GetBitmap(), _('Home page'), None)
                                               ):
            if label is None:
                self.toolbar.AddSeparator()
            else:
                self.toolbar.AddTool(_id, label, bmp, shorthelp)
                if handler:
                    self.Bind(wx.EVT_TOOL, handler, id=_id)
                _id += 10

        for _id in range(10, 61, 10):
            self.toolbar.EnableTool(_id, False)

        self.toolbar.Realize()

        self.client = wx.Panel(self)
        self.images = [None] * 4
        sizer = wx.GridSizer(2, 2, 1, 1)
        self.bitmaps = [wx.StaticBitmap(self.client) for i in range(4)]
        sizer.AddMany(self.bitmaps)
        self.client.SetSizer(sizer)

        self.client.Bind(wx.EVT_SIZE, self.OnClientSize)

        self.status = self.CreateStatusBar(1)

        self.Connect(-1, -1, EVT_NEW_IMAGE, self.OnSetImage)

        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.cap = None
        self.running = threading.Event()
        self.stream_thread = threading.Thread(target=self.VideoStreamThread)
        self.stream_thread.start()

    def OnClose(self, event):
        self.running.set()
        self.stream_thread.join()
        if self.cap:
            self.cap.release()
            self.cap = None
        event.Skip()

    def OnClientSize(self, event):
        for i in range(4):
            self.RefreshImage(i)
        event.Skip()

    def VideoStreamThread(self):
        while not self.running.is_set():
            if self.cap:
                flag, img = self.cap.read()
                if flag:
                    wx.PostEvent(self, NewImageEvent(0, img))
            else:
                img = pyautogui.screenshot()
                wx.PostEvent(self, NewImageEvent(0, img))
            time.sleep(0.1)

    def OnVideoSourceChanged(self, event):
        if self.cap:
            self.cap.release()
            self.cap = None
        index = event.GetSelection()
        if index > 0:
            self.cap = cv2.VideoCapture(index - 1)

    def OnSetImage(self, event):
        img = event.image
        if isinstance(img, Image.Image):
            w, h = event.image.size
            img = wx.ImageFromBuffer(w, h, img.tobytes())
        elif isinstance(img, numpy.ndarray):
            h, w = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = wx.ImageFromBuffer(w, h, img)
        self.images[event.index] = img
        self.RefreshImage(event.index)

    def RefreshImage(self, index):
        def rescale(img, w, h):
            ow, oh = img.GetSize()
            if w / h > ow / oh:
                new_w = int(h / oh * ow)
                new_h = h
            else:
                new_w = w
                new_h = int(w / ow * oh)
            return img.Copy().Rescale(new_w, new_h, wx.IMAGE_QUALITY_HIGH)

        img = self.images[index]
        if img is None:
            bmp = wx.NullBitmap
        else:
            windowsize = self.client.GetSize()
            w, h = windowsize.x // 2, windowsize.y // 2
            img = rescale(img, w, h)
            bmp = wx.Bitmap(img)
        self.bitmaps[index].SetBitmap(bmp)


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
