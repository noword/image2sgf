pyinstaller img2sgf_gui.pyw ^
    --onefile^
    --exclude-module PyQt5^
    --exclude-module PyQt6^
    --exclude-module PySide6^
    --exclude-module pandas^
    --exclude-module matplotlib^
    --exclude-module torch.distributions^
    --exclude-module torchaudio^
    --exclude-module IPython^
    --bootloader-ignore-signals

pyinstall img2sgf.py ^
    --onefile^
    --exclude-module PyQt5^
    --exclude-module PyQt6^
    --exclude-module PySide6^
    --exclude-module pandas^
    --exclude-module matplotlib^
    --exclude-module torch.distributions^
    --exclude-module torchaudio^
    --exclude-module IPython^
    --bootloader-ignore-signals