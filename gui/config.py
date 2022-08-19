from collections import UserDict
import json
import os
import wx


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
