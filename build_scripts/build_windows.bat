@echo off
echo === Starting Build for Windows ===
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

pyinstaller --noconfirm --onefile --windowed ^
    --name "RootLocus_Windows" ^
    --paths "src" ^
    --icon "assets\icon.ico" ^
    --hidden-import "latex" ^
    --hidden-import "PIL._tkinter_finder" ^
    --collect-all "customtkinter" ^
    --add-data "assets;assets" ^
    src\interface.py

echo Success! The executable is in dist\RootLocus_Windows.exe
pause
