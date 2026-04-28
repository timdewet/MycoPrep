# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for MycoPrep.

Build a desktop bundle:

    pip install pyinstaller
    pyinstaller packaging/mycoprep.spec

Produces ``dist/MycoPrep/`` containing the MycoPrep executable and
supporting DLLs / data files. Zip and distribute the whole folder. The
first launch downloads cellpose model weights to the user's
``~/.cellpose`` directory (a few hundred MB) — needs internet on first
run only.

PyInstaller does NOT cross-compile. To produce a Windows .exe, run the
above command on a Windows machine.
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules


datas: list = []
binaries: list = []
hiddenimports: list = []

# Project assets — bundled at the runtime root so ``resource_root()``
# (in ``mycoprep/gui/_resources.py``) finds them.
datas.append(("../assets/logo/logo.svg", "logo"))

# ``collect_all`` walks each package's files (Python modules, package
# data, DLLs) so the bundle contains everything ``mycoprep.*`` imports
# at runtime.
for pkg in (
    "mycoprep",
    "cellpose",
    "torch",
    "torchvision",
    "pylibCZIrw",
    "tifffile",
    "aicspylibczi",
    "skimage",
    "h5py",
    "pyqtgraph",
):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

for pkg in ("skimage", "scipy", "mycoprep"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        pass

excludes = [
    "tkinter",
    "matplotlib.tests",
    "numpy.tests",
    "pandas.tests",
    "scipy.tests",
    "skimage.tests",
    "pytest",
    "_pytest",
    "IPython",
    "notebook",
    "jupyter",
    "sphinx",
    "test",
]


a = Analysis(
    ["../src/mycoprep/__main__.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)


pyz = PYZ(a.pure, a.zipped_data)


# A windowed (no console) executable. Provide a ``logo/logo.ico`` if you
# want a custom Windows icon — convert the SVG to ICO once and drop it
# in ``assets/logo/``, then point ``icon`` at it.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MycoPrep",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)


coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MycoPrep",
)
