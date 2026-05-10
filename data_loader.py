"""
===============================================================
Step 1 & 2 — Dataset Loading via FiftyOne / HuggingFace
---------------------------------------------------------------
Loads the MVTec-AD dataset from the Voxel51 Hub, filters to
the selected categories, and saves organised train/test splits
as PNG images on disk for downstream PatchCore training.
===============================================================
"""

import os
import shutil
import logging
from pathlib import Path

# ------------------------------------------------------------------ #
#  HuggingFace Authentication & Cache Redirect                         #
#  (redirects away from the locked system cache)                       #
# ------------------------------------------------------------------ #
_HF_TOKEN = "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"
os.environ["HUGGING_FACE_HUB_TOKEN"] = _HF_TOKEN
os.environ["HF_TOKEN"]               = _HF_TOKEN

# Use a local writable cache inside the project folder
_LOCAL_CACHE = str(Path("./hf_cache").resolve())
os.environ["HF_HOME"]              = _LOCAL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(_LOCAL_CACHE) / "hub")
os.environ["HF_DATASETS_CACHE"]    = str(Path(_LOCAL_CACHE) / "datasets")
os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(Path(_LOCAL_CACHE) / "fiftyone_zoo")
Path(_LOCAL_CACHE).mkdir(parents=True, exist_ok=True)

import fiftyone as fo
import fiftyone.utils.huggingface as fouh
from PIL import Image
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

CATEGORIES   = CFG["dataset"]["categories"]
DATA_DIR     = Path(CFG["dataset"]["data_dir"])
IMAGE_SIZE   = CFG["dataset"]["image_size"]


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
def save_sample(sample: fo.Sample, dest: Path, size: int) -> None:
    """Copy / resize one FiftyOne sample image to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(sample.filepath).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img.save(dest)


def export_category(dataset: fo.Dataset, category: str) -> None:
    """
    Filter the FiftyOne dataset to one category and export
    train (good) / test (good + defective) splits to disk.

    Actual MVTec-AD field schema (Voxel51/mvtec-ad on HuggingFace):
      - category  : Classification  → label = category name (e.g. 'metal_nut')
      - defect    : Classification  → label = 'good' | <defect_type>
      - split     : StringField     → 'train' | 'test'
    """
    log.info(f"Exporting category: {category}")

    # Filter by the 'category' Classification field
    cat_view = dataset.filter_labels(
        "category", fo.ViewField("label") == category
    )

    if len(cat_view) == 0:
        log.warning(f"No samples found for category '{category}'. Skipping.")
        return

    log.info(f"  Found {len(cat_view)} samples for '{category}'")

    for sample in cat_view.iter_samples(progress=True):
        split = sample.split if sample.split else "train"

        # defect.label is 'good' for normal images, defect name otherwise
        defect_label = "good"
        if sample.defect is not None and sample.defect.label:
            defect_label = sample.defect.label

        fname = Path(sample.filepath).name
        dest  = DATA_DIR / category / split / defect_label / fname
        save_sample(sample, dest, IMAGE_SIZE)

    log.info(f"  ✓ {category} exported to {DATA_DIR / category}")


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def load_and_prepare_dataset() -> fo.Dataset:
    """
    Load MVTec-AD from HuggingFace Hub via FiftyOne and export
    selected categories to a flat directory tree.
    """
    log.info("Loading MVTec-AD from Voxel51 HuggingFace Hub …")
    log.info("(This may take several minutes on first run.)")

    dataset = fouh.load_from_hub(
        "Voxel51/mvtec-ad",
        name="mvtec-ad",       # FiftyOne forbids '/' in dataset names
        overwrite=True,        # replace if a previous partial load exists
    )

    log.info(f"Dataset loaded — {len(dataset)} total samples")
    log.info(f"Split values : {dataset.distinct('split')}")
    log.info(f"Categories  : {dataset.distinct('category.label')}")
    log.info(f"Defects     : {dataset.distinct('defect.label')[:20]}")

    # Export selected categories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        export_category(dataset, category)

    log.info("Dataset preparation complete.")
    return dataset


if __name__ == "__main__":
    dataset = load_and_prepare_dataset()

    # Optionally launch the FiftyOne App for visual inspection
    session = fo.launch_app(dataset)
    session.wait()
