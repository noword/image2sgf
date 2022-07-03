import wx
import os
import imgs
from PIL import Image, ImageDraw, ImageFont
from img2sgf import get_models, get_board_image, classifier_board, NpBoxPostion, DEFAULT_IMAGE_SIZE, get_sgf
from img2sgf.sgf2img import GameImageGenerator, GetAllThemes
import threading
import numpy as np


_ = wx.GetTranslation

EVT_NEW_IMAGE = wx.NewId()


class NewImageEvent(wx.PyEvent):
    def __init__(self, index, image):
        super().__init__()
        self.SetEventType(EVT_NEW_IMAGE)
        self.index = index
        self.image = image


class Model:
    def __init__(self, window, board_path='board.pth', stone_path='stone.pth'):
        self.window = window
        self.board_model = self.stone_model = None

        def load_model(board_path, stone_path):
            self.board_model, self.stone_model = get_models(board_path, stone_path)

        threading.Thread(target=load_model, args=(board_path, stone_path)).start()

    def recognize(self, img_path):
        if self.board_model is None or self.stone_model is None:
            return False

        img = Image.open(img_path)
        wx.PostEvent(self.window, NewImageEvent(0, img.copy()))
        try:
            _img, boxes, scores = get_board_image(self.board_model, img)
        except BaseException as err:
            print(err)
            return False

        box_image = img.copy()
        draw = ImageDraw.ImageDraw(box_image)
        y0, y1 = boxes[0, [1, 3]]
        font_size = int(y1 - y0)
        font = ImageFont.truetype('arial.ttf', size=font_size)
        for i, box in enumerate(boxes):
            draw.rectangle(box.tolist(), outline='green', width=5)
            draw.text((box[0], box[1] - font_size), f'{scores[i]:.2f}', fill='red', font=font)

        wx.PostEvent(self.window, NewImageEvent(1, box_image))

        _img, boxes, scores = get_board_image(self.board_model, _img)
        self.board = classifier_board(self.stone_model, _img)

        self.board_image = Image.fromarray(_img)
        wx.PostEvent(self.window, NewImageEvent(2, self.__get_board_image_with_stones(self.board_image, self.board)))

        self.sgf = get_sgf(self.board)
        wx.PostEvent(self.window, NewImageEvent(3, self.__get_board_image_from_sgf(self.sgf)))
        return True

    def __get_board_image_from_sgf(self, sgf):
        TMP_SGF = 'tmp.sgf'
        open(TMP_SGF, 'wb').write(sgf.serialise())
        gig = GameImageGenerator(GetAllThemes()['real-stones'])
        sgf_image = gig.get_game_image(TMP_SGF)
        os.remove(TMP_SGF)
        return sgf_image

    def __get_board_image_with_stones(self, board_image, board):
        board_image = board_image.copy()
        draw = ImageDraw.ImageDraw(board_image)
        box_pos = NpBoxPostion(width=DEFAULT_IMAGE_SIZE, size=19)
        for y in range(19):
            for x in range(19):
                color = board[x][y] >> 1
                _sides, _color = ((None, None),
                                  (3, 'green'),
                                  (4, 'blue')
                                  )[color]

                if _sides is not None:
                    _x, _y = box_pos._grid_pos[x][y]
                    r = int(box_pos.grid_size / 3)
                    for i in range(r - 5, r):
                        draw.regular_polygon((_x, _y, i), _sides, outline=_color)
        return board_image

    def rotate(self, clockwise=True):
        if clockwise:
            self.board_image = self.board_image.rotate(270)
            self.board = np.rot90(self.board)
        else:
            self.board_image = self.board_image.rotate(90)
            self.board = np.rot90(self.board, 3)

        wx.PostEvent(self.window, NewImageEvent(2, self.__get_board_image_with_stones(self.board_image, self.board)))
        self.sgf = get_sgf(self.board)
        wx.PostEvent(self.window, NewImageEvent(3, self.__get_board_image_from_sgf(self.sgf)))


class MainFrame(wx.Frame):
    def __init__(self, parent, title):
        super(MainFrame, self).__init__(parent,
                                        title=title,
                                        size=(840, 480))
        self.Locale = wx.Locale()
        self.Locale.AddCatalogLookupPathPrefix('locale')
        self.Locale.AddCatalog('messages')
        self.SetIcon(imgs.GO.GetIcon())
        self.Center()

        self.toolbar = self.CreateToolBar(wx.TB_FLAT)
        tsize = (32, 32)

        self.toolbar.SetToolBitmapSize(tsize)
        self.toolbar.AddTool(10,
                             _('Screenshot'),
                             imgs.SCREENSHOT.GetBitmap(),
                             _('Capture a screeshot'))
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
        self.toolbar.AddTool(70,
                             _('About'),
                             imgs.HELP.GetBitmap(),
                             _('About'))

        self.toolbar.EnableTool(30, False)
        self.toolbar.EnableTool(40, False)
        self.toolbar.EnableTool(50, False)
        self.toolbar.Realize()

        self.client = wx.Panel(self)

        self.client.SetBackgroundColour(wx.WHITE)
        self.images = [None] * 4
        sizer = wx.GridSizer(2, 2, 1, 1)
        self.bitmaps = [wx.StaticBitmap(self.client) for i in range(4)]
        sizer.AddMany(self.bitmaps)
        self.client.SetSizer(sizer)

        self.client.Bind(wx.EVT_SIZE, self.OnClientSize)

        self.status = wx.StatusBar(self)
        self.SetStatusBar(self.status)

        self.model = Model(self)
        self.Connect(-1, -1, EVT_NEW_IMAGE, self.OnSetImage)

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
        if img is not None:
            windowsize = self.client.GetSize()
            w, h = windowsize.x // 2, windowsize.y // 2
            img = rescale(img, w, h)
            self.bitmaps[index].SetBitmap(wx.Bitmap(img))

    def OnSetImage(self, event):
        img = event.image
        if isinstance(img, Image.Image):
            w, h = event.image.size
            img = wx.ImageFromBuffer(w, h, img.tobytes())
        self.images[event.index] = img
        self.RefreshImage(event.index)

    def OnOpenClick(self, event):
        def recognize(path):
            if self.model.recognize(path):
                self.toolbar.EnableTool(30, True)
                self.toolbar.EnableTool(40, True)
                self.toolbar.EnableTool(50, True)
            else:
                dlg = wx.MessageDialog(self, _('Recognition failed'),
                                       _('Error'),
                                       wx.OK | wx.ICON_INFORMATION
                                       #wx.YES_NO | wx.NO_DEFAULT | wx.CANCEL | wx.ICON_INFORMATION
                                       )
                dlg.ShowModal()
                dlg.Destroy()
            self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))

        dlg = wx.FileDialog(self,
                            message=_('Choose a file'),
                            defaultDir=os.getcwd(),
                            defaultFile='',
                            wildcard='|'.join(['pictures|*.jpeg;* .png;*.jpg;*.bmp',
                                               'All files (*.*)|*.*']),
                            style=wx.FD_OPEN | wx.FD_CHANGE_DIR | wx.FD_FILE_MUST_EXIST | wx.FD_PREVIEW
                            )
        if dlg.ShowModal() == wx.ID_OK:
            self.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
            threading.Thread(target=recognize, args=(dlg.GetPath(), )).start()

    def OnSaveClick(self, event):
        dlg = wx.FileDialog(self,
                            message=_('Save file as ...'), defaultDir=os.getcwd(),
                            defaultFile='',
                            wildcard='SGF (*.sgf)|*.sgf',
                            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
                            )
        if dlg.ShowModal() == wx.ID_OK:
            open(dlg.GetPath(), 'wb').write(self.model.sgf.serialise())

    def OnRotateClick(self, event):
        def rotate(clockwise):
            self.model.rotate(clockwise)
            self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))

        self.SetCursor(wx.Cursor(wx.CURSOR_WAIT))
        threading.Thread(target=rotate, args=(event.GetId() == 50,)).start()


class App(wx.App):
    def InitLocale(self):
        import sys
        if sys.platform.startswith('win') and sys.version_info > (3, 8):
            import locale
            locale.setlocale(locale.LC_ALL, "C")


def run():
    app = App(True, 'img2sgf.log')
    frame = MainFrame(None, title='img2sgf')
    frame.Show()
    app.MainLoop()


if __name__ == '__main__':
    run()
