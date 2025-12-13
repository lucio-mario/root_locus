#!/bin/bash
set -e
echo "=== Starting Build for Linux ==="
rm -rf build/ dist/

pyinstaller --noconfirm --onefile --windowed \
    --name "RootLocus_Linux" \
    --paths "src" \
    --hidden-import "latex" \
    --hidden-import "PIL._tkinter_finder" \
    --collect-all "customtkinter" \
    --add-data "assets:assets" \
    src/interface.py

echo "Success! The binary is in dist/RootLocus_Linux"
