#!/bin/bash
set -e
echo "=== Iniciando Build para Linux ==="
rm -rf build/ dist/

pyinstaller --noconfirm --onefile --windowed \
    --name "RootLocus_Linux" \
    --paths "src" \
    --hidden-import "latex" \
    --hidden-import "PIL._tkinter_finder" \
    --collect-all "customtkinter" \
    --add-data "assets:assets" \
    src/interface.py

echo "✅ Sucesso! O binário está em dist/RootLocus_Linux"
