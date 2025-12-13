#!/bin/bash

# Garante que o script pare se houver erro
set -e

echo "=== Iniciando Build para Linux ==="

# Limpa builds anteriores para garantir um executável limpo
rm -rf build/ dist/

# Comando do PyInstaller
# Nota: Estamos rodando assumindo que você está na raiz do projeto.
# - Separador de arquivos no Linux é ":"
# - --add-data "assets:assets" copia a pasta assets local para dentro do executável
# - --paths "src" ajuda o PyInstaller a achar os módulos na pasta src

pyinstaller --noconfirm --onefile --windowed \
    --name "RootLocus_Linux" \
    --paths "src" \
    --hidden-import "latex" \
    --add-data "assets:assets" \
    src/interface.py

echo "✅ Sucesso! O binário está em dist/RootLocus_Linux"
