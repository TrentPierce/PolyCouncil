# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None

# Get absolute path to icon file
spec_dir = os.path.dirname(os.path.abspath(SPEC))
icon_path = os.path.abspath(os.path.join(spec_dir, 'PolyCouncilIco.ico'))
if not os.path.exists(icon_path):
    icon_path = None  # Fallback if icon not found
    print(f"Warning: Icon file not found at {icon_path}")
else:
    print(f"Using icon: {icon_path}")
    # Verify icon is readable
    try:
        with open(icon_path, 'rb') as f:
            header = f.read(6)
            if header[:2] != b'\x00\x00' or header[2:4] != b'\x01\x00':
                print(f"Warning: Icon file may not be valid ICO format")
    except Exception as e:
        print(f"Warning: Could not verify icon file: {e}")

a = Analysis(
    ['council.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'aiohttp',
        'aiohttp.client',
        'aiohttp.connector',
        'aiohttp.helpers',
        'aiohttp.http_parser',
        'aiohttp.streams',
        'aiohttp.typedefs',
        'aiohttp.web',
        'qdarktheme',
        'sqlite3',
        'requests',
        'asyncio',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PolyCouncil',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disabled - can interfere with icon embedding
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_path,  # Application icon
)

