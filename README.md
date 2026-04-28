# MycoPrep

Bacterial microscopy pre-processing pipeline with a PyQt6 desktop GUI.

MycoPrep wraps focus picking, FOV split, Cellpose-SAM segmentation, and a cell-quality classifier into a single wizard-style application for fungal/bacterial specimen prep.

This repo merges the former **ImagingPipeline** (core library) and **ImagePipelineGUI** (PyQt6 frontend) projects into a single Python package: `mycoprep` with `mycoprep.core` (library) and `mycoprep.gui` (GUI) subpackages.

## Install

### For end-users (Windows)

1. Download the latest `MycoPrep-windows.zip` from the [Releases](https://github.com/timdewet/MycoPrep/releases) page.
2. Unzip anywhere (e.g. `C:\Program Files\MycoPrep\`).
3. Double-click `MycoPrep.exe`.

First launch downloads ~few hundred MB of Cellpose model weights to `%USERPROFILE%\.cellpose\` — needs internet once. Subsequent launches are offline. No Python install required.

### For developers (from source)

Works on macOS, Linux, and Windows.

```bash
git clone https://github.com/timdewet/MycoPrep.git
cd MycoPrep
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .[test]
```

Requires Python 3.10+. PyTorch + Cellpose are heavy installs; first install may take a while.

## Run

| Command | Purpose |
| --- | --- |
| `mycoprep` | Launch the GUI |
| `mycoprep-cli --help` | Top-level CLI (was `imagingpipeline`) |
| `mycoprep-focus --help` | Focus-picking CLI (was `focuspicker`) |
| `python -m mycoprep` | Equivalent to `mycoprep` |

## Layout

```
src/mycoprep/
├── core/    # library: focus, segmentation, classification, CZI handling
└── gui/     # PyQt6 wizard: panels, pipeline runner, live preview
assets/
├── logo/        # app icon
└── models_mtb/  # trained classifier weights (*.pth, ~1.5 MB each)
data/
└── labeled_data/  # training crops + labels
packaging/
└── mycoprep.spec  # PyInstaller spec for desktop bundle
design_review/
└── CRITIQUE.md    # post-launch UX backlog
```

## Building a desktop bundle

The bundle config lives in [packaging/mycoprep.spec](packaging/mycoprep.spec). PyInstaller does **not** cross-compile — build on the OS you intend to ship to.

### Windows (`.exe`)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
pip install pyinstaller
pyinstaller packaging\mycoprep.spec
```

Output: `dist\MycoPrep\` containing `MycoPrep.exe` plus DLLs/data. Zip the **whole folder** to distribute.

For GPU (CUDA) inference, install the matching PyTorch wheel before `pip install -e .`:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU-only is fine if you don't need GPU; the bundle is much smaller.

### macOS / Linux

```bash
pip install pyinstaller
pyinstaller packaging/mycoprep.spec
```

Output: `dist/MycoPrep/`. Runs only on the OS family it was built on.

### Building Windows from a non-Windows host

PyInstaller can't cross-compile. Use one of:

- **GitHub Actions** with `runs-on: windows-latest` (recommended for repeatable releases).
- A Windows VM (Parallels, UTM, VMware) on your Mac.
- A borrowed/spare Windows machine.

## Releases

Pre-built Windows bundles are published on the [Releases](https://github.com/timdewet/MycoPrep/releases) page. Each release ships a zipped `dist\MycoPrep\` folder — see the end-user install steps above.

## License

MIT — see [LICENSE](LICENSE).
