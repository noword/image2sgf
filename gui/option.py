import wx
from img2sgf.sgf2img import GetAllThemes, Theme, GameImageGenerator

_ = wx.GetTranslation


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
        gig = GameImageGenerator(Theme(theme))
        sgf_image = gig.get_game_image(b'(;GM[1]FF[4]KM[6.5]SZ[19]AB[pd][dp]AW[pp][dd])')
        img = sgf_image.resize((480, 480))
        self.bmp.SetBitmap(wx.Bitmap.FromBuffer(*img.size, img.tobytes()))
