# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['img2sgf_gui.pyw'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PyQt6', 'PySide6', 'pandas', 'matplotlib', 'torch.distributions', 'torchaudio', 'IPython', 'tcl', 'tcl8', 'tk', 'scipy', 'FixTk', '_tkinter', 'tkinter', 'Tkinter'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

EXCLUDES = ['_C.cp38-win_amd64', '_C_flatbuffer.cp38-win_amd64']
for d in a.datas:
    for e in EXCLUDES:
        if e in d[0] or e in d[1]:
            a.datas.remove(d)


pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='img2sgf_gui',
    debug=False,
    bootloader_ignore_signals=True,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='go.ico',
)
