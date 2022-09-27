#! /usr/bin/env python3
# coding=utf-8
import json
import os
import pathlib


class Theme:
    def __init__(self, name=None):
        if name:
            self.load(name)

    def load(self, name):
        if not os.path.exists(name):
            name = f'./themes/{name}/theme.json'
        self._theme = json.load(open(name, 'r', encoding='utf-8'))
        _path = os.path.split(name)[0]
        for key in ('black', 'white', 'board', 'font'):
            if key in self._theme:
                value = self._theme[key]
                if isinstance(value, list):
                    for i in range(len(value)):
                        value[i] = os.path.join(_path, value[i])
                else:
                    self._theme[key] = os.path.join(_path, value)

        if 'font' not in self._theme:
            self._theme['font'] = './themes/NotoSansMono-Regular.ttf'

        if 'scaling_ratio' not in self._theme:
            self._theme['scaling_ratio'] = 1

        if 'adjust_ratio' not in self._theme:
            self._theme['adjust_ratio'] = 0

        if 'bold_border' not in self._theme:
            self._theme['bold_border'] = 0

    def __getitem__(self, key):
        return self._theme.get(key)


def GetAllThemes(path='./themes'):
    themes = {}
    for theme_path in pathlib.Path(path).glob('*/theme.json'):
        theme_path = str(theme_path)
        name = theme_path.split(os.sep)[-2]
        themes[name] = Theme(theme_path)
    return themes
