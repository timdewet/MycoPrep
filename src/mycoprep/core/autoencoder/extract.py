"""Extract 512-d embeddings from trained autoencoder and produce strain profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import AutoencoderConfig
from .dataset import HDF5CropDataset
from .train import _collate_crops

ProgressCB = Callable[[float, str], None]


def extract_embeddings(
    h5_paths: Sequence[Path],
    model_path: Path,
    output_dir: Path,
    config: AutoencoderConfig,
    *,
    batch_size: int = 512,
    device: Optional[str] = None,
    progress_cb: Optional[ProgressCB] = None,
    # Kept for back-compat — Harmony is now applied at viz time, not here.
    # Setting this no longer changes the saved embeddings.
    harmony_correct: bool = True,
    run_ids_per_file: Optional[Sequence[str]] = None,
) -> Path:
    """Run trained encoder on all crops to produce embedding matrices.

    Outputs:
        - ``<output_dir>/cnn_embeddings.parquet`` — per-cell embeddings
        - ``<output_dir>/strain_profiles.csv`` — per-condition mean profiles

    Returns:
        Path to cnn_embeddings.parquet.
    """
    from .train import _detect_device

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _detect_device(device)

    if progress_cb is None:
        progress_cb = lambda f, m: None

    progress_cb(0.0, "Loading model...")
    model = config.build_model()
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    progress_cb(0.05, "Loading crops...")
    dataset = HDF5CropDataset(
        h5_paths,
        target_size=config.crop_size,
        in_channels=config.in_channels,
        image_channels=config.image_channels,
        include_mask=config.include_mask,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, len(h5_paths)),
        collate_fn=_collate_crops,
        pin_memory=(device != "cpu"),
    )

    all_embeddings: list[np.ndarray] = []
    all_cell_uids: list[str] = []
    all_conditions: list[str] = []
    all_fov_ids: list[int] = []
    all_file_indices: list[int] = []
    all_genes: list[str] = []
    all_drugs: list[str] = []
    all_concentrations: list[str] = []
    all_is_drug: list[bool] = []
    all_is_control: list[bool] = []

    n_batches = len(loader)
    progress_cb(0.10, f"Extracting embeddings from {len(dataset)} cells...")

    with torch.no_grad():
        for batch_idx, (crops, metas) in enumerate(loader):
            crops = crops.to(device)
            z = model.encode(crops)  # (B, latent_dim)
            all_embeddings.append(z.cpu().numpy())

            for m in metas:
                all_cell_uids.append(m.get("cell_uid", ""))
                all_conditions.append(m.get("condition_label", ""))
                all_fov_ids.append(m.get("fov_id", -1))
                all_file_indices.append(m.get("file_idx", 0))
                all_genes.append(m.get("gene", ""))
                all_drugs.append(m.get("drug", ""))
                all_concentrations.append(m.get("concentration", ""))
                all_is_drug.append(bool(m.get("is_drug", False)))
                all_is_control.append(bool(m.get("is_control", False)))

            if (batch_idx + 1) % max(1, n_batches // 20) == 0:
                frac = 0.10 + 0.70 * (batch_idx + 1) / n_batches
                progress_cb(frac, f"Batch {batch_idx + 1}/{n_batches}")

    dataset.close()

    embeddings = np.vstack(all_embeddings)  # (N, latent_dim)
    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]

    # Build per-cell DataFrame
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, "cell_uid", all_cell_uids)
    df.insert(1, "condition_label", all_conditions)
    df.insert(2, "fov_id", all_fov_ids)

    # Per-cell run_id. The file's stem is unreliable as a run identifier
    # because every library run's consolidated crops file is conventionally
    # named ``all_crops.h5`` — so two different runs would collide. Prefer
    # the explicit ``run_ids_per_file`` mapping when the caller provides it.
    if run_ids_per_file is not None and len(run_ids_per_file) == len(h5_paths):
        run_ids = [str(run_ids_per_file[fi]) for fi in all_file_indices]
    else:
        run_ids = [str(h5_paths[fi].stem) for fi in all_file_indices]
    df.insert(3, "run_id", run_ids)

    # Per-cell gene / drug / control labels straight from the H5 metadata
    # (populated by the labelling stage). Lets downstream OT analysis
    # rank conditions cleanly without re-deriving from condition_label.
    df.insert(4, "gene", all_genes)
    df.insert(5, "drug", all_drugs)
    df.insert(6, "concentration", all_concentrations)
    df.insert(7, "is_drug", all_is_drug)
    df.insert(8, "is_control", all_is_control)

    # NB: Harmony batch correction is no longer applied here. We save raw
    # encoder output and let the visualisation layer apply Harmony on demand
    # (toggleable from the Analysis tab). This keeps the saved embeddings
    # canonical and reusable across different correction strategies.
    emb_path = output_dir / "cnn_embeddings.parquet"
    df.to_parquet(emb_path, index=False)
    progress_cb(0.90, f"Saved {len(df)} cell embeddings (raw)")

    # Compute strain-level profiles (mean per condition).
    # Save both raw and centered variants. Centering subtracts the global
    # per-cell mean before aggregation, removing the dominant "average cell"
    # axis that the autoencoder learns; without this, per-condition means
    # collapse to ~1% spread in cosine sim and clustering looks meaningless.
    # Centered profiles are what should be used for similarity / UMAP.
    progress_cb(0.92, "Computing strain profiles...")
    profiles = df.groupby("condition_label")[emb_cols].mean()
    profiles.to_csv(output_dir / "strain_profiles.csv")

    # Centered profiles: subtract the global per-cell mean.
    global_mean = df[emb_cols].mean()
    centered = df.copy()
    centered[emb_cols] = df[emb_cols].values - global_mean.values
    centered_profiles = centered.groupby("condition_label")[emb_cols].mean()
    centered_profiles.to_csv(output_dir / "strain_profiles_centered.csv")

    # NT-relative profiles: subtract the negative-control centroid. This is
    # the standard "treatment vs wild-type" representation used in
    # image-based profiling. Picks any condition whose first token starts
    # with "NT" or "WT" or "DMSO" as a control; falls back to global mean.
    profile_genes = profiles.index.to_series().str.split().str[0].str.upper()
    is_control = profile_genes.str.match(r"^(NT|WT|DMSO|CTRL|UNT)\d*$")
    if is_control.any():
        anchor = profiles[is_control].mean().values
        nt_relative = profiles - anchor
        nt_relative.to_csv(output_dir / "strain_profiles_nt_relative.csv")
        msg = f"Saved {len(profiles)} strain profiles (raw + centered + NT-relative)"
    else:
        msg = f"Saved {len(profiles)} strain profiles (raw + centered; no controls detected)"
    progress_cb(0.95, msg)

    # cnn_embeddings.parquet is the raw output now; the separate
    # cnn_embeddings_raw.parquet was redundant under the old "always save raw
    # alongside corrected" scheme and is no longer written.

    progress_cb(1.0, "Embedding extraction complete.")
    return emb_path


def _harmony_correct(embeddings: np.ndarray, run_ids: np.ndarray) -> np.ndarray:
    """Apply Harmony batch correction on embeddings by run_id.

    Note: kept for back-compat. Production batch correction now happens at
    the per-profile level inside ``render_embeddings_html`` rather than
    inside extract_embeddings — see comments in that module.
    Orientation handling is delegated to ``qc_plots._run_harmony_oriented``
    so the result is always (n_obs, n_features) regardless of the
    installed harmonypy version (0.0.x returns transposed; 2.x doesn't).
    """
    try:
        from mycoprep.core.extract.qc_plots import _run_harmony_oriented
        import numpy as np
        unique = np.unique(np.asarray(run_ids).astype(str))
        return _run_harmony_oriented(
            embeddings, run_ids,
            nclust=min(max(2, len(unique)), 5),
        )
    except ImportError:
        return embeddings
    except Exception:
        return embeddings


