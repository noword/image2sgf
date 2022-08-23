import wx
import threading
import imgs
from PIL import Image
import pyautogui
import os
import time
import webbrowser
from .config import Config
from .option import OptionDialog
from img2sgf import get_board_model, get_stone_model, get_board_image, classifier_board, get_sgf, NpBoxPostion, DEFAULT_IMAGE_SIZE
from img2sgf.sgf2img import GameImageGenerator, Theme, GetAllThemes
import numpy
from .widgets import ImagePanel

_ = wx.GetTranslation

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
        self.config = Config('img2sgf_gui.json')
        self.Locale = wx.Locale(self.config['language'])
        self.Locale.AddCatalogLookupPathPrefix('locale')
        self.Locale.AddCatalog('messages')
        self.SetIcon(imgs.GO.GetIcon())
        self.Center()

        self.toolbar = self.CreateToolBar(wx.TB_FLAT)
        self.toolbar.SetToolBitmapSize((32, 32))

        _id = 10
        for label, bmp, shorthelp, handler in ((_('Screenshot'), imgs.SCREENSHOT.GetBitmap(), _('Capture a screeshot'), self.OnCaptureScreen),
                                               (None, None, None, None),
                                               (_('Open'), imgs.OPEN.GetBitmap(), _('Open a picture'), self.OnOpenClick),
                                               (_('Save'), imgs.SAVE.GetBitmap(), _('Save the sgf file'), self.OnSaveClick),
                                               (None, None, None, None),
                                               (_('Left'), imgs.LEFT.GetBitmap(), _('Rotate left'), self.OnRotateClick),
                                               (_('Right'), imgs.RIGHT.GetBitmap(), _('Rotate right'), self.OnRotateClick),
                                               (None, None, None, None),
                                               (_('Option'), imgs.OPTIONS.GetBitmap(), _('Option'), self.OnOptionClick),
                                               (_('Home'), imgs.HOME.GetBitmap(), _('Home page'), self.OnHomeClick)
                                               ):
            if label is None:
                self.toolbar.AddSeparator()
            else:
                self.toolbar.AddTool(_id, label, bmp, shorthelp)
                if handler:
                    self.Bind(wx.EVT_TOOL, handler, id=_id)
                _id += 10

        for _id in range(10, 51, 10):
            self.toolbar.EnableTool(_id, False)

        self.toolbar.Realize()

        self.client = wx.Panel(self)

        self.client.SetBackgroundColour(wx.WHITE)

        self.splitter = wx.SplitterWindow(self.client)
        self.right_splitter = wx.SplitterWindow(self.splitter)
        self.original_panel = ImagePanel(self.splitter)
        self.board_panel = ImagePanel(self.right_splitter)
        self.sgf_panel = ImagePanel(self.right_splitter)
        self.panels = [self.original_panel, self.board_panel, self.sgf_panel]
        # self.bitmaps = [wx.StaticBitmap(self.splitter),
        #                 wx.StaticBitmap(self.right_splitter),
        #                 wx.StaticBitmap(self.right_splitter)]

        self.right_splitter.SplitHorizontally(self.board_panel, self.sgf_panel)
        self.splitter.SplitVertically(self.original_panel, self.right_splitter)
        self.right_splitter.SetSashGravity(0.5)
        self.splitter.SetSashGravity(0.5)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.splitter, 1, wx.EXPAND)
        self.client.SetSizer(sizer)

        # self.client.Bind(wx.EVT_SIZE, self.OnClientSize)
        # self.splitter.Bind(wx.EVT_SPLITTER_SASH_POS_CHANGED, self.OnClientSize)
        # self.right_splitter.Bind(wx.EVT_SPLITTER_SASH_POS_CHANGED, self.OnClientSize)

        self.status = self.CreateStatusBar(1)

        self.__LoadModels()
        self.Connect(-1, -1, EVT_NEW_IMAGE, self.OnSetImage)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

    def __LoadModels(self):
        self.board_model = self.stone_model = None

        def load_model():
            for name in ('board.pth', 'stone.pth'):
                if not os.path.exists(name):
                    dlg = wx.MessageDialog(self, name + _(' is missing'),
                                           _('Error'),
                                           wx.OK | wx.ICON_INFORMATION
                                           # wx.YES_NO | wx.NO_DEFAULT | wx.CANCEL | wx.ICON_INFORMATION
                                           )
                    dlg.ShowModal()
                    dlg.Destroy()
                    return

            self.status.SetStatusText(_('loading models'))
            self.board_model = get_board_model()
            self.board_model.eval()
            self.stone_model = get_stone_model()
            self.stone_model.eval()
            self.toolbar.EnableTool(10, True)
            self.toolbar.EnableTool(20, True)
            self.status.SetStatusText('')

        threading.Thread(target=load_model).start()

    def __Recognize(self, img):
        if self.board_model is None or self.stone_model is None:
            return False

        if isinstance(img, str):
            img = Image.open(img).convert('RGB')

        wx.PostEvent(self, NewImageEvent(0, img.copy()))
        wx.PostEvent(self, NewImageEvent(1, None))
        wx.PostEvent(self, NewImageEvent(2, None))
        self.status.SetStatusText(_('step 1: detect 4 corners of board'))

        try:
            _img, boxes, scores = get_board_image(self.board_model, img)
        except BaseException as err:
            print(err)
            try:
                _img, boxes, scores = get_board_image(self.board_model, img, False)
            except BaseException as err:
                self.status.SetStatusText(_("Error: Can't identify the board."))
                print(err)
                return False

        wx.PostEvent(self, NewImageEvent(0, self.__GetBoxImage(img, boxes, scores)))
        self.status.SetStatusText(_('step 2: perspective correct the board, then classify stones'))

        if min(scores) < 0.7:
            _img0, boxes0, scores0 = get_board_image(self.board_model, _img)
            if sum(scores0) > sum(scores):
                _img, boxes, scores = _img0, boxes0, scores0
        self.board = classifier_board(self.stone_model, _img)
        # self.board = classifier_board_kmeans(_img)

        self.board_image = _img
        wx.PostEvent(self, NewImageEvent(1, self.__GetBoardImageWithStones(self.board_image, self.board)))
        self.status.SetStatusText(_('step 3: generating sgf'))

        self.sgf = get_sgf(self.board)

        try:
            img = self.__GetBoardImageFromSgf(self.sgf, self.config['theme'])
        except BaseException as err:
            print(err)
            self.status.SetStatusText(_("Error: Can't generate sgf."))
            return False

        wx.PostEvent(self, NewImageEvent(2, img))
        self.status.SetStatusText(_('All done. You can save the sgf now.'))
        return True

    def __GetBoxImage(self, img, boxes, scores):
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

    def __GetBoardImageFromSgf(self, sgf, theme):
        gig = GameImageGenerator(Theme(theme))
        sgf_image = gig.get_game_image(sgf.serialise())
        return sgf_image

    def __GetBoardImageWithStones(self, board_image, board):
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

    def __Rotate(self, clockwise=True):
        if clockwise:
            self.board_image = self.board_image.rotate(270)
            self.board = numpy.rot90(self.board)
        else:
            self.board_image = self.board_image.rotate(90)
            self.board = numpy.rot90(self.board, 3)

        wx.PostEvent(self, NewImageEvent(1, self.__GetBoardImageWithStones(self.board_image, self.board)))
        self.sgf = get_sgf(self.board)
        wx.PostEvent(self, NewImageEvent(2, self.__GetBoardImageFromSgf(self.sgf, self.config['theme'])))

    def OnClose(self, event):
        self.config.save()
        event.Skip()

    def OnSetImage(self, event):
        self.panels[event.index].SetImage(event.image)

    def Recognize(self, img):
        [self.toolbar.EnableTool(i, False) for i in range(10, 60, 10)]
        if self.__Recognize(img):
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
                open(dlg.GetPath(), 'wb').write(self.sgf.serialise())

    def OnRotateClick(self, event):
        def rotate(clockwise):
            self.__Rotate(clockwise)
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
        with OptionDialog(self.config.copy(), self, -1, _('Option')) as dlg:
            dlg.CenterOnParent()
            if dlg.ShowModal() == wx.ID_OK:
                if self.config['theme'] != dlg.config['theme'] and not self.sgf_panel.IsEmpty():
                    img = self.__GetBoardImageFromSgf(self.sgf, dlg.config['theme'])
                    wx.PostEvent(self, NewImageEvent(2, img))
                self.config = dlg.config
