@echo off
echo === Iniciando Build para Windows ===

:: Limpa pastas antigas
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

:: Comando do PyInstaller
:: Nota: Separador de arquivos no Windows é ";"
:: Inclui o ícone .ico especificamente para o executável

pyinstaller --noconfirm --onefile --windowed ^
    --name "RootLocus_Windows" ^
    --paths "src" ^
    --icon "assets\icon.ico" ^
    --hidden-import "latex" ^
    --add-data "assets;assets" ^
    src\interface.py

echo ✅ Sucesso! O executável esta em dist\RootLocus_Windows.exe
pause
