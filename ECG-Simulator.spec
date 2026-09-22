# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for ECG Simulator.

Used by both the local Windows build and the GitHub Actions release workflow,
so the two produce the same thing.

    pyinstaller ECG-Simulator.spec --noconfirm

Windows gets a single self-contained .exe.  macOS gets a .app bundle built from
a one-directory layout, which is the arrangement macOS actually expects and is
far less prone to Gatekeeper and code-signing trouble than a one-file binary.
"""

import sys

APP_NAME = "ECG-Simulator"
MAC = sys.platform == "darwin"

# scipy is used only by the test suite, never by the app - and it is by far the
# largest dependency.  The Qt modules below are pulled in by the PyQt6 hooks but
# nothing here touches them.
EXCLUDES = [
    "scipy",
    "matplotlib",
    "tkinter",
    "PIL",
    "pytest",
    "IPython",
    "PyQt6.QtWebEngineCore",
    "PyQt6.QtWebEngineWidgets",
    "PyQt6.QtQml",
    "PyQt6.QtQuick",
    "PyQt6.QtQuick3D",
    "PyQt6.QtMultimedia",
    "PyQt6.QtMultimediaWidgets",
    "PyQt6.QtBluetooth",
    "PyQt6.QtNetworkAuth",
    "PyQt6.QtPositioning",
    "PyQt6.QtSerialPort",
    "PyQt6.QtSql",
    "PyQt6.QtTest",
    "PyQt6.QtDesigner",
    "PyQt6.QtHelp",
    "PyQt6.QtPdf",
    "PyQt6.QtPdfWidgets",
]

icon = None
if not MAC and sys.platform.startswith("win"):
    from pathlib import Path
    candidate = Path("packaging/icon.ico")
    if candidate.exists():
        icon = str(candidate)

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)
pyz = PYZ(a.pure)

if MAC:
    # One-directory layout, wrapped in a .app bundle.
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        strip=False,
        upx=False,
        console=False,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name=APP_NAME,
    )
    app = BUNDLE(
        coll,
        name="ECG Simulator.app",
        icon=None,
        bundle_identifier="io.github.goga-lanchava.ecg-simulator",
        info_plist={
            "CFBundleName": "ECG Simulator",
            "CFBundleDisplayName": "ECG Simulator",
            "CFBundleShortVersionString": "1.1.0",
            "CFBundleVersion": "1.1.0",
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
else:
    # Single self-contained executable.
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        runtime_tmpdir=None,
        console=False,
        icon=icon,
    )
