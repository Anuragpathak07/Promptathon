"""
===============================================================
Training Script — train PatchCore on all selected categories
---------------------------------------------------------------
Usage:
    python train.py                         # all categories in config
    python train.py --categories metal_nut  # single category
===============================================================
"""

import argparse
import logging
import yaml

from model import build_patchcore

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)


def train_category(category: str) -> None:
    log.info(f"━━━ Training PatchCore: {category} ━━━")
    pc = build_patchcore(category)
    pc.fit(
        batch_size   = CFG["training"]["batch_size"],
        num_workers  = 0,   # 0 = required on Windows (avoids spawn deadlocks)
    )
    log.info(f"━━━ Done: {category} ━━━\n")


def main():
    parser = argparse.ArgumentParser(description="Train PatchCore models")
    parser.add_argument(
        "--categories", nargs="+",
        default=CFG["dataset"]["categories"],
        help="Space-separated list of MVTec categories to train"
    )
    args = parser.parse_args()

    log.info(f"Categories to train: {args.categories}")
    for cat in args.categories:
        train_category(cat)

    log.info("All categories trained successfully.")


if __name__ == "__main__":
    main()