# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

spec_dir = os.path.dirname(os.path.abspath(SPEC))

datas = []
binaries = []
hiddenimports = [
    "sklearn.utils._cython_blas",
    "sklearn.neighbors._typedefs",
    "sklearn.neighbors._quad_tree",
    "sklearn.tree._utils",
    "PIL._tkinter_finder",
]

for pkg in ("numpy", "matplotlib", "sklearn", "seaborn"):
    tmp_ret = collect_all(pkg)
    datas += tmp_ret[0]
    binaries += tmp_ret[1]
    hiddenimports += tmp_ret[2]

logo_file = os.path.join(spec_dir, "logo emsi.png")
if os.path.isfile(logo_file):
    datas.append((logo_file, "."))

assets_dir = os.path.join(spec_dir, "src", "assets")
if os.path.isdir(assets_dir):
    for name in os.listdir(assets_dir):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
            datas.append((os.path.join(assets_dir, name), os.path.join("src", "assets")))

a = Analysis(
    ["APP.py"],
    pathex=[spec_dir],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="APP",
    debug=False,
    bootloader_ignore_signals=False,
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
)
