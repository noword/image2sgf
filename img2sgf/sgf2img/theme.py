#! /usr/bin/env python3
# coding=utf-8
import json
import os
import pathlib
from dataclasses import *
from typing import List


@dataclass
class Theme:
    black: List = field(default_factory=list)
    white: List = field(default_factory=list)
    board: str = ''
    board_resize: bool = True
    line_color: str = 'black'
    font: str = './themes/NotoSansMono-Regular.ttf'
    scaling_ratio: float = 1.
    adjust_ratio: float = 0.
    bold_border: float = 0.
    json_name: InitVar = None

    def __post_init__(self, json_name):
        if json_name is not None:
            _theme = json.load(open(json_name, 'r', encoding='utf-8'))
            _path = os.path.split(json_name)[0]
            for key in ('black', 'white', 'board', 'font'):
                if key in _theme:
                    value = _theme[key]
                    if isinstance(value, list):
                        for i in range(len(value)):
                            value[i] = os.path.join(_path, value[i])
                    else:
                        _theme[key] = os.path.join(_path, value)

            for k, v in _theme.items():
                setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)


def GetAllThemes(path='./themes'):
    themes = {}
    for theme_path in pathlib.Path(path).glob('*/theme.json'):
        theme_path = str(theme_path)
        name = theme_path.split(os.sep)[-2]
        themes[name] = Theme(json_name=theme_path)
    return themes
