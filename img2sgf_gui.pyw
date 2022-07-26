import wx
import os
import imgs
from PIL import Image
from img2sgf import get_models, get_board_image, classifier_board, NpBoxPostion, DEFAULT_IMAGE_SIZE, get_sgf
from img2sgf.sgf2img import GameImageGenerator, Theme, GetAllThemes
import threading
import numpy as np
import pyautogui
import time
import webbrowser
import json
from collections import UserDict
import tempfile

_ = wx.GetTranslation

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
            self.window.toolbar.EnableTool(20, True)
            self.window.status.SetStatusText('')

        threading.Thread(target=load_model, args=(board_path, stone_path)).start()

    def recognize(self, img):
        if self.board_model is None or self.stone_model is None:
            return False

        if isinstance(img, str):
            img = Image.open(img).convert('RGB')

        wx.PostEvent(self.window, NewImageEvent(0, img.copy()))
        wx.PostEvent(self.window, NewImageEvent(1, None))
        wx.PostEvent(self.window, NewImageEvent(2, None))
        wx.PostEvent(self.window, NewImageEvent(3, None))
        self.window.status.SetStatusText(_('step 1: detect 4 corners of board'))

        try:
            _img, boxes, scores = get_board_image(self.board_model, img)
        except BaseException as err:
            print(err)
            try:
                _img, boxes, scores = get_board_image(self.board_model, img, False)
            except BaseException as err:
                self.window.status.SetStatusText(_("Error: Can't identify the board."))
                print(err)
                return False

        wx.PostEvent(self.window, NewImageEvent(1, self.__get_box_image(img, boxes, scores)))
        self.window.status.SetStatusText(_('step 2: perspective correct the board, then classify stones'))

        if min(scores) < 0.7:
            _img0, boxes0, scores0 = get_board_image(self.board_model, _img)
            if sum(scores0) > sum(scores):
                _img, boxes, scores = _img0, boxes0, scores0
        self.board = classifier_board(self.stone_model, _img)

        self.board_image = _img
        wx.PostEvent(self.window, NewImageEvent(2, self.__get_board_image_with_stones(self.board_image, self.board)))
        self.window.status.SetStatusText(_('step 3: generating sgf'))

        self.sgf = get_sgf(self.board)

        try:
            img = self.__get_board_image_from_sgf(self.sgf, self.theme)
        except BaseException as err:
            print(err)
            self.window.status.SetStatusText(_("Error: Can't generate sgf."))
            return False

        wx.PostEvent(self.window, NewImageEvent(3, img))
        self.window.status.SetStatusText(_('All done. You can save the sgf now.'))

        return True

    def __get_box_image(self, img, boxes, scores):
        w, h = img.size
        bmp = wx.Bitmap.FromBuffer(w, h, img.tobytes())
        dc = wx.MemoryDC(bmp)

        y0, y1 = boxes[0, [1, 3]]
        font_size = int(y1 - y0)
        font = wx.Font()
        font.SetPixelSize(wx.Size(0, font_size))
        dc.SetFont(font)
        rects = [[box[0], box[1], box[2] - box[0], box[3] - box[1]] for box in boxes]
        dc.DrawRectangleList(rects, wx.Pen('green', width=font_size // 5), wx.Brush('green', wx.TRANSPARENT))

        dc.SetTextForeground('red')
        for i, box in enumerate(boxes.astype(int)):
            dc.DrawText(f'{scores[i]:.2f}', (box[0], box[1] - font_size))
        return bmp.ConvertToImage()

    def __get_board_image_from_sgf(self, sgf, theme):
        TMP_SGF = tempfile.gettempdir() + '/tmp.sgf'
        open(TMP_SGF, 'wb').write(sgf.serialise())
        gig = GameImageGenerator(Theme(theme))
        sgf_image = gig.get_game_image(TMP_SGF)
        os.remove(TMP_SGF)
        return sgf_image

    def __get_board_image_with_stones(self, board_image, board):
        w, h = board_image.size
        bmp = wx.Bitmap.FromBuffer(w, h, board_image.tobytes())
        dc = wx.MemoryDC(bmp)
        box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
        # rects = []
        # for _boxes in box_pos:
        #     for box in _boxes:
        #         rects.append((*box[:2], box_pos.grid_size, box_pos.grid_size))
        # dc.DrawRectangleList(rects, wx.Pen('violet'), wx.TRANSPARENT_BRUSH)

        shape_size = int(box_pos.grid_size / 3)
        half_shape_size = shape_size // 2
        black_shapes = []
        white_shapes = []
        for y in range(19):
            for x in range(19):
                color = board[x][y] >> 1
                if color == 0:
                    continue
                elif color == 1:
                    _x, _y = box_pos._grid_pos[x][y]
                    black_shapes.append((_x - half_shape_size, _y - half_shape_size, shape_size, shape_size))
                else:  # color == 2
                    _x, _y = box_pos._grid_pos[x][y]
                    white_shapes.append((_x - half_shape_size, _y - half_shape_size, shape_size, shape_size))

        dc.DrawEllipseList(black_shapes, wx.Pen('green', 5), wx.TRANSPARENT_BRUSH)
        dc.DrawRectangleList(white_shapes, wx.Pen('blue', 5), wx.TRANSPARENT_BRUSH)
        return bmp.ConvertToImage()

    def rotate(self, clockwise=True):
        if clockwise:
            self.board_image = self.board_image.rotate(270)
            self.board = np.rot90(self.board)
        else:
            self.board_image = self.board_image.rotate(90)
            self.board = np.rot90(self.board, 3)

        wx.PostEvent(self.window, NewImageEvent(2, self.__get_board_image_with_stones(self.board_image, self.board)))
        self.sgf = get_sgf(self.board)
        wx.PostEvent(self.window, NewImageEvent(3, self.__get_board_image_from_sgf(self.sgf, self.theme)))


class Config(UserDict):
    def __init__(self, name):
        self.name = name
        if os.path.exists(name):
            self.load(name)
        else:
            self.data = {'theme': 'real-stones',
                         'language': wx.LANGUAGE_DEFAULT}

    def load(self, name):
        self.data = json.load(open(name))

    def save(self, name=None):
        if name is None:
            name = self.name
        json.dump(self.data, open(name, 'w'))


class OptionDialog(wx.Dialog):
    LANGUAGES = {wx.LANGUAGE_DEFAULT: 'System Default',
                 wx.LANGUAGE_CHINESE_SIMPLIFIED: wx.Locale.GetLanguageName(wx.LANGUAGE_CHINESE_SIMPLIFIED),
                 wx.LANGUAGE_ENGLISH: wx.Locale.GetLanguageName(wx.LANGUAGE_ENGLISH)
                 }

    def __init__(self, config, *args, **kw):
        super().__init__(*args, **kw)

        self.config = config
        gbsizer = wx.GridBagSizer(vgap=5, hgap=5)

        gbsizer.Add(wx.StaticText(self, -1, _('Language')), (1, 1), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTRE_VERTICAL)

        lang_combobox = wx.ComboBox(self,
                                    -1,
                                    self.LANGUAGES[config['language']],
                                    choices=list(self.LANGUAGES.values()),
                                    style=wx.CB_READONLY)
        lang_combobox.Bind(wx.EVT_COMBOBOX, self.OnLangChanged)
        gbsizer.Add(lang_combobox, (1, 2))

        gbsizer.Add(wx.StaticText(self, -1, _('Theme')), (2, 1), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTRE_VERTICAL)
        theme_combobox = wx.ComboBox(self,
                                     -1,
                                     config['theme'],
                                     choices=list(GetAllThemes().keys()),
                                     style=wx.CB_READONLY
                                     )
        theme_combobox.Bind(wx.EVT_COMBOBOX, self.OnThemeChanged)

        gbsizer.Add(theme_combobox, (2, 2))

        self.bmp = wx.StaticBitmap(self, -1)
        self.__set_theme_image(config['theme'])
        gbsizer.Add(self.bmp, (3, 2), flag=wx.ALL | wx.EXPAND, border=10)

        btnsizer = wx.StdDialogButtonSizer()
        btnsizer.AddButton(wx.Button(self, wx.ID_OK, _('OK')))
        btnsizer.AddButton(wx.Button(self, wx.ID_CANCEL, _('Cancel')))
        btnsizer.Realize()
        gbsizer.Add(btnsizer, (4, 2), flag=wx.ALIGN_RIGHT | wx.ALL | wx.EXPAND, border=10)

        self.SetSizer(gbsizer)
        self.Layout()
        self.Fit()

    def OnThemeChanged(self, event):
        theme = event.GetString()
        self.__set_theme_image(theme)
        self.config['theme'] = theme

    def OnLangChanged(self, event):
        d = {v: k for k, v in OptionDialog.LANGUAGES.items()}
        self.config['language'] = d.get(event.GetString(), wx.LANGUAGE_DEFAULT)

    def __set_theme_image(self, theme):
        TMP_SGF = tempfile.gettempdir() + '/tmp.sgf'
        open(TMP_SGF, 'w').write('(;GM[1]FF[4]KM[6.5]SZ[19]AB[pd][dp]AW[pp][dd])')
        gig = GameImageGenerator(Theme(theme))
        sgf_image = gig.get_game_image(TMP_SGF)
        os.remove(TMP_SGF)
        w, h = self.GetSize()
        img = sgf_image.resize((480, 480))
        self.bmp.SetBitmap(wx.Bitmap.FromBuffer(*img.size, img.tobytes()))


class MainFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MainFrame, self).__init__(parent,
                                        title=title,
                                        size=(840, 480))
        self.config = Config('img2sgf_gui.json')
        self.Locale = wx.Locale(self.config['language'])
        self.Locale.AddCatalogLookupPathPrefix('locale')
        self.Locale.AddCatalog('messages')
        self.SetIcon(imgs.GO.GetIcon())
        self.Center()

        self.toolbar = self.CreateToolBar(wx.TB_FLAT)
        self.toolbar.SetToolBitmapSize((32, 32))
        self.toolbar.AddTool(10,
                             _('Screenshot'),
                             imgs.SCREENSHOT.GetBitmap(),
                             _('Capture a screeshot'))
        self.Bind(wx.EVT_TOOL, self.OnCaptureScreen, id=10)

        self.toolbar.AddSeparator()

        self.toolbar.AddTool(20,
                             _('Open'),
                             imgs.OPEN.GetBitmap(),
                             _('Open a picture'))
        self.Bind(wx.EVT_TOOL, self.OnOpenClick, id=20)

        self.toolbar.AddTool(30,
                             _('Save'),
                             imgs.SAVE.GetBitmap(),
                             _('Save the sgf file'))
        self.Bind(wx.EVT_TOOL, self.OnSaveClick, id=30)

        self.toolbar.AddSeparator()

        self.toolbar.AddTool(40,
                             _('Left'),
                             imgs.LEFT.GetBitmap(),
                             _('Rotate left'))
        self.Bind(wx.EVT_TOOL, self.OnRotateClick, id=40)

        self.toolbar.AddTool(50,
                             _('Right'),
                             imgs.RIGHT.GetBitmap(),
                             _('Rotate right'))
        self.Bind(wx.EVT_TOOL, self.OnRotateClick, id=50)

        self.toolbar.AddSeparator()

        self.toolbar.AddTool(60,
                             _('Option'),
                             imgs.OPTIONS.GetBitmap(),
                             _('Option'),
                             )
        self.Bind(wx.EVT_TOOL, self.OnOptionClick, id=60)

        self.toolbar.AddTool(70,
                             _('Home'),
                             imgs.HOME.GetBitmap(),
                             _('Home page'))
        self.Bind(wx.EVT_TOOL, self.OnHomeClick, id=70)

        for id in range(10, 51, 10):
            self.toolbar.EnableTool(id, False)

        self.toolbar.Realize()

        self.client = wx.Panel(self)

        self.client.SetBackgroundColour(wx.WHITE)
        self.images = [None] * 4
        sizer = wx.GridSizer(2, 2, 1, 1)
        self.bitmaps = [wx.StaticBitmap(self.client) for i in range(4)]
        sizer.AddMany(self.bitmaps)
        self.client.SetSizer(sizer)

        self.client.Bind(wx.EVT_SIZE, self.OnClientSize)

        self.status = self.CreateStatusBar(1)

        self.model = Model(self, theme=self.config['theme'])
        self.Connect(-1, -1, EVT_NEW_IMAGE, self.OnSetImage)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def OnClose(self, event):
        self.config.save()
        event.Skip()

    def OnClientSize(self, event):
        for i in range(4):
            self.RefreshImage(i)
        event.Skip()

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

    def OnSetImage(self, event):
        img = event.image
        if isinstance(img, Image.Image):
            w, h = event.image.size
            img = wx.ImageFromBuffer(w, h, img.tobytes())
        self.images[event.index] = img
        self.RefreshImage(event.index)

    def Recognize(self, img):
        [self.toolbar.EnableTool(i, False) for i in range(10, 60, 10)]
        if self.model.recognize(img):
            [self.toolbar.EnableTool(i, True) for i in range(10, 60, 10)]
        else:
            [self.toolbar.EnableTool(i, True) for i in range(10, 30, 10)]
            dlg = wx.MessageDialog(self, _('Recognition failed'),
                                   _('Error'),
                                   wx.OK | wx.ICON_INFORMATION
                                   # wx.YES_NO | wx.NO_DEFAULT | wx.CANCEL | wx.ICON_INFORMATION
                                   )
            dlg.ShowModal()
            dlg.Destroy()
        self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))

    def OnOpenClick(self, event):
        with wx.FileDialog(self,
                           message=_('Choose a file'),
                           defaultDir=os.getcwd(),
                           defaultFile='',
                           wildcard='|'.join(['pictures(*.jpeg;* .png;*.jpg;*.bmp)|*.jpeg;* .png;*.jpg;*.bmp',
                                              'All files (*.*)|*.*']),
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_PREVIEW
                           ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                self.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
                threading.Thread(target=self.Recognize, args=(dlg.GetPath(), )).start()

    def OnSaveClick(self, event):
        with wx.FileDialog(self,
                           message=_('Save file as ...'), defaultDir=os.getcwd(),
                           defaultFile='',
                           wildcard='SGF (*.sgf)|*.sgf',
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
                           ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                open(dlg.GetPath(), 'wb').write(self.model.sgf.serialise())

    def OnRotateClick(self, event):
        def rotate(clockwise):
            self.model.rotate(clockwise)
            self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))

        self.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
        threading.Thread(target=rotate, args=(event.GetId() == 50,)).start()

    def OnCaptureScreen(self, event):
        def CaptureScreen():
            font = wx.Font()
            font.SetPixelSize(wx.Size(0, 32))
            small_font = wx.Font()
            small_font.SetPixelSize(wx.Size(0, 16))

            for i in range(3, 0, -1):
                bmp = wx.Bitmap(512, 256)
                dc = wx.MemoryDC(bmp)

                dc.SetBackground(wx.Brush("white"))
                dc.Clear()

                dc.SetFont(small_font)
                dc.DrawText(_('make sure only one board on the screen'), 10, 10)

                dc.SetFont(font)
                dc.DrawText(str(i), 240, 112)

                del(dc)

                img = bmp.ConvertToImage()
                wx.PostEvent(self, NewImageEvent(0, img))
                time.sleep(1)

            img = pyautogui.screenshot()
            self.Recognize(img)
            self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))

        self.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
        [self.toolbar.EnableTool(i, False) for i in range(10, 60, 10)]
        threading.Thread(target=CaptureScreen).start()

    def OnHomeClick(self, event):
        webbrowser.open('https://github.com/noword/image2sgf')

    def OnOptionClick(self, event):
        with OptionDialog(self.config, self, -1, _('Option')) as dlg:
            dlg.CenterOnParent()
            if dlg.ShowModal() == wx.ID_OK:
                self.config = dlg.config
                self.model.theme = self.config['theme']


class App(wx.App):
    def InitLocale(self):
        import sys
        if sys.platform.startswith('win') and sys.version_info > (3, 8):
            import locale
            locale.setlocale(locale.LC_ALL, "C")


def run():
    app = App(True, 'img2sgf.log')
    frame = MainFrame(None, title='img2sgf v0.05')
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    run()
