# MycoPrep

Bacterial microscopy pre-processing pipeline with a PyQt6 desktop GUI.

MycoPrep wraps focus picking, FOV split, Cellpose-SAM segmentation, and a cell-quality classifier into a single wizard-style application for fungal/bacterial specimen prep.

This repo merges the former **ImagingPipeline** (core library) and **ImagePipelineGUI** (PyQt6 frontend) projects into a single Python package: `mycoprep` with `mycoprep.core` (library) and `mycoprep.gui` (GUI) subpackages.

## Install

```bash
git clone https://github.com/<user>/MycoPrep.git
cd MycoPrep
python -m venv .venv
source .venv/bin/activate
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

```bash
pyinstaller packaging/mycoprep.spec
```

## License

MIT — see [LICENSE](LICENSE).
