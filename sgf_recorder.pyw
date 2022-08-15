import wx
from img2sgf import get_board_model, get_stone_model, get_board_position, get_board_image, classifier_board, NpBoxPostion, DEFAULT_IMAGE_SIZE, get_sgf
from img2sgf.sgf2img import GameImageGenerator, Theme, GetAllThemes, GridPosition
import recorder_imgs
import cv2
import pyautogui
from PIL import Image
import threading
import time
import numpy
from sgfmill import sgf
import torch
import torchvision

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


class SgfRecorder:
    def __init__(self, model):
        self.stone_model = model
        self.box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
        self.grid_pos = GridPosition(width=DEFAULT_IMAGE_SIZE, size=19)
        self.sgf = sgf.Sgf_game(size=19)
        self.last_img = None

    def refresh(self, img):
        img = to_cv2(img)
        if self.last_img is None:
            board = self._classify(self._splite_img(img)).reshape(19, 19)
            blacks = []
            whites = []
            for y in range(19):
                for x in range(19):
                    color = board[x][y] >> 1
                    if color == 1:
                        blacks.append([x, y])
                    elif color == 2:
                        whites.append([x, y])
            self.sgf.get_root().set_setup_stones(blacks, whites)
        else:

            poses = []
            for y in range(19):
                for x in range(19):
                    _x, _y = self.grid_pos[y][x]
                    if cv2.norm(img[_y, _x], self.last_img[_y, _x]) > 0.1:
                        poses.append((y, x))
            if 0 < len(poses) < 19 * 8:
                imgs = []
                for y, x in poses:
                    x0, y0, x1, y1 = self.box_pos[y][x].astype(int)
                    imgs.append(torchvision.transforms.ToTensor()(img[y0:y1, x0:x1]))

                results = self._classify(torch.stack(imgs))
                print([r >> 1 for r in results])
                print(poses)
        self.last_img = img

    def _classify(self, imgs):
        return self.stone_model(imgs).argmax(1)

    def _splite_img(self, img):
        img = torchvision.transforms.ToTensor()(img)
        imgs = torch.empty((19 * 19, 3, int(self.box_pos.grid_size), int(self.box_pos.grid_size)))
        for y in range(19):
            for x in range(19):
                x0, y0, x1, y1 = self.box_pos[y][x].astype(int)
                imgs[x + y * 19] = img[:, y0:y1, x0:x1]
        return imgs


EVT_NEW_IMAGE = wx.NewIdRef()


class NewImageEvent(wx.PyEvent):
    def __init__(self, index, image):
        super().__init__()
        self.SetEventType(EVT_NEW_IMAGE)
        self.index = index
        self.image = image


class MainFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MainFrame, self).__init__(parent,
                                        title=title,
                                        size=(840, 480))
        self.transform = None

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
        for label, bmp, shorthelp, handler in ((_('Detect'), recorder_imgs.DETECT.GetBitmap(), _('Detect the board'), self.OnDetectClick),
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

        threading.Thread(target=self._LoadModel, args=('board.pth', 'stone.pth')).start()

    def _LoadModel(self, board_path, stone_path):
        self.status.SetStatusText(_('loading models'))
        self.board_model = get_board_model(board_path)
        self.board_model.eval()
        self.stone_model = get_stone_model(stone_path)
        self.stone_model.eval()
        self.sgf_recorder = SgfRecorder(self.stone_model)
        self.toolbar.EnableTool(10, True)
        self.status.SetStatusText('')

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
            time.sleep(0.2)

    def OnVideoSourceChanged(self, event):
        if self.cap:
            self.cap.release()
            self.cap = None
        index = event.GetSelection()
        if index > 0:
            self.cap = cv2.VideoCapture(index - 1)

    def OnSetImage(self, event):
        img = event.image
        img = to_wx(img)
        self.images[event.index] = img
        self.RefreshImage(event.index)

        if event.index == 0 and self.transform is not None:
            img = cv2.warpPerspective(img_wx_to_cv2(img), self.transform, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))
            wx.PostEvent(self, NewImageEvent(1, img))
        elif event.index == 1:
            self.sgf_recorder.refresh(img)

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

    def OnDetectClick(self, event):
        self.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
        self.toolbar.EnableTool(10, False)
        self.toolbar.EnableTool(20, False)
        self.toolbar.EnableTool(50, False)
        self.toolbar.EnableTool(60, False)

        img = img_wx_to_pil(self.images[0])
        try:
            self.boxes, self.scores = get_board_position(self.board_model, img, True)
        except BaseException:
            self.transform = self.boxes = self.scores = None

        if self.boxes is not None:
            startpoints = self.boxes[:, :2].tolist()
            box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
            endpoints = [box_pos[18][0][:2],  # top left
                         box_pos[18][18][:2],  # top right
                         box_pos[0][0][:2],  # bottom left
                         box_pos[0][18][:2]  # bottom right
                         ]
            self.transform = cv2.getPerspectiveTransform(numpy.array(startpoints, numpy.float32),
                                                         numpy.array(endpoints, numpy.float32))
            self.toolbar.EnableTool(20, True)
            self.toolbar.EnableTool(50, True)
            self.toolbar.EnableTool(60, True)
            dlg = wx.MessageDialog(self, _("THe board has been detected, don't move the camera any more"),
                                   _('Notice'),
                                   wx.OK | wx.ICON_INFORMATION
                                   # wx.YES_NO | wx.NO_DEFAULT | wx.CANCEL | wx.ICON_INFORMATION
                                   )
            dlg.ShowModal()
            dlg.Destroy()
        else:
            self.images[1] = None
            self.RefreshImage(1)
            dlg = wx.MessageDialog(self, _('Failed to discover any board'),
                                   _('Error'),
                                   wx.OK | wx.ICON_INFORMATION
                                   # wx.YES_NO | wx.NO_DEFAULT | wx.CANCEL | wx.ICON_INFORMATION
                                   )
            dlg.ShowModal()
            dlg.Destroy()

        self.toolbar.EnableTool(10, True)
        self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))

        # print(self.boxes, self.scores)


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
