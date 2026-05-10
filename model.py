"""
===============================================================
Step 3 — PatchCore Feature Extractor (ResNet-50 backbone)
---------------------------------------------------------------
Implements the PatchCore algorithm:
  1. Extract patch-level features from ResNet-50 intermediate
     layers (layer2 + layer3) for all *normal* training images.
  2. Sub-sample the resulting coreset via greedy approximate
     k-centre coverage (random sampling for speed).
  3. At inference, score a test image by the maximum nearest-
     neighbour distance between its patch features and the
     coreset memory bank.
  4. Produce a spatial anomaly score map (heatmap).
===============================================================
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml
from tqdm import tqdm
import faiss

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}")


# ================================================================== #
#  Dataset                                                            #
# ================================================================== #
class MVTecDataset(Dataset):
    """
    Flat-directory MVTec dataset loader.

    Directory layout (produced by data_loader.py):
        data/mvtec/<category>/train/good/*.png
        data/mvtec/<category>/test/good/*.png
        data/mvtec/<category>/test/<defect>/*.png
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        category: str,
        split: str = "train",
        image_size: int = 224,
        data_dir: str = "./data/mvtec",
    ):
        self.split      = split
        self.image_size = image_size

        root = Path(data_dir) / category / split
        if not root.exists():
            raise FileNotFoundError(
                f"Data directory not found: {root}\n"
                "Run data_loader.py first to download and export the dataset."
            )

        self.samples: List[Tuple[Path, int]] = []  # (path, label)
        for defect_dir in sorted(root.iterdir()):
            label = 0 if defect_dir.name == "good" else 1
            for img_path in sorted(defect_dir.glob("*.*")):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.samples.append((img_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

        self.transform = T.Compose([
            T.Resize((image_size, image_size), Image.Resampling.LANCZOS),
            T.ToTensor(),
            T.Normalize(mean=self.MEAN, std=self.STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, str(path)


# ================================================================== #
#  Feature Extractor                                                   #
# ================================================================== #
class PatchCoreFeatureExtractor(nn.Module):
    """
    ResNet-50 with hooked intermediate layers.
    Returns concatenated patch-level features from layer2 & layer3.
    """

    def __init__(
        self,
        layers: Sequence[str] = ("layer2", "layer3"),
        pretrained: bool = True,
    ):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Remove classification head
        self.feature_layers = nn.ModuleDict()
        self._hooks          = []
        self._features: Dict[str, torch.Tensor] = {}

        # Register forward hooks
        for name, module in backbone.named_children():
            self.feature_layers[name] = module
            if name in layers:
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )

        self.target_layers = list(layers)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._features.clear()
        _ = self._run_backbone(x)
        return {k: self._features[k] for k in self.target_layers}

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        # Stop after the last target layer — no need to run avgpool / fc,
        # and doing so causes a shape error because ResNet flattens inside
        # its own forward() (not as a child module).
        last_target = self.target_layers[-1]
        for name, module in self.feature_layers.items():
            x = module(x)
            if name == last_target:
                break
        return x


# ================================================================== #
#  PatchCore                                                          #
# ================================================================== #
class PatchCore:
    """
    PatchCore anomaly detector for one MVTec category.

    Parameters
    ----------
    category              : MVTec category name
    patch_size            : local neighbourhood aggregation (avg-pool kernel)
    num_neighbors         : k for k-NN anomaly scoring
    coreset_sampling_ratio: fraction of train patches kept in memory bank
    layers                : ResNet-50 layers to hook
    image_size            : spatial input resolution
    data_dir              : root of exported dataset
    output_dir            : where to save the trained model
    """

    def __init__(
        self,
        category: str,
        patch_size: int                = 3,
        num_neighbors: int             = 9,
        coreset_sampling_ratio: float  = 0.1,
        layers: Sequence[str]          = ("layer2", "layer3"),
        image_size: int                = 224,
        data_dir: str                  = "./data/mvtec",
        output_dir: str                = "./outputs/models",
    ):
        self.category               = category
        self.patch_size             = patch_size
        self.num_neighbors          = num_neighbors
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.image_size             = image_size
        self.data_dir               = data_dir
        self.output_dir             = Path(output_dir) / category
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = PatchCoreFeatureExtractor(
            layers=list(layers), pretrained=True
        ).to(DEVICE)
        self.extractor.eval()

        # Adaptive average pooling for local patch aggregation
        self.adaptive_pool = nn.AvgPool2d(
            kernel_size=patch_size, stride=1, padding=patch_size // 2
        )

        self.memory_bank: Optional[np.ndarray] = None
        self.index:       Optional[faiss.Index] = None
        self.threshold:   float = 0.0

    # -------------------------------------------------------------- #
    #  Feature extraction helpers                                      #
    # -------------------------------------------------------------- #
    def _extract_features(
        self, dataloader: DataLoader, subsample: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Extract and concatenate multi-scale patch features for all
        images in *dataloader*. Returns (feature_matrix, labels).
        """
        all_features: List[np.ndarray] = []
        all_labels:   List[int]         = []

        with torch.no_grad():
            for imgs, labels, _ in tqdm(dataloader, desc="Extracting features"):
                imgs = imgs.to(DEVICE)
                feat_map = self._get_patch_features(imgs)
                # feat_map: (B, C, H', W')
                B, C, H, W = feat_map.shape
                # Reshape to (B*H*W, C)
                patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
                
                if subsample and self.coreset_sampling_ratio < 1.0:
                    n_keep = max(1, int(len(patches) * self.coreset_sampling_ratio))
                    idx = np.random.choice(len(patches), n_keep, replace=False)
                    patches = patches[idx]
                    
                all_features.append(patches)
                all_labels.extend(labels.tolist())

        return np.concatenate(all_features, axis=0), all_labels


    def _get_patch_features(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass → hook intermediate features → pool → concatenate.
        Returns: (B, C_total, H', W')
        """
        feature_maps = self.extractor(imgs)

        # Upsample all to the size of the largest map, then concatenate
        target_size = feature_maps[self.extractor.target_layers[0]].shape[-2:]

        resized = []
        for key in self.extractor.target_layers:
            fm = feature_maps[key]
            if fm.shape[-2:] != target_size:
                fm = nn.functional.interpolate(
                    fm, size=target_size, mode="bilinear", align_corners=False
                )
            # Local patch aggregation
            fm = self.adaptive_pool(fm)
            resized.append(fm)

        concat = torch.cat(resized, dim=1)               # (B, C_total, H', W')
        # L2-normalise across channel dim
        concat = nn.functional.normalize(concat, p=2, dim=1)
        return concat

    # -------------------------------------------------------------- #
    #  Training (coreset construction)                                 #
    # -------------------------------------------------------------- #
    def fit(self, batch_size: int = 32, num_workers: int = 0) -> None:
        if sys.platform == "win32" and num_workers > 0:
            log.warning("Windows detected: forcing num_workers=0 to prevent DataLoader hangs.")
            num_workers = 0

        """
        Build the memory bank from normal (good) training images.
        """
        log.info(f"[{self.category}] Building PatchCore memory bank …")

        train_ds = MVTecDataset(
            self.category, split="train",
            image_size=self.image_size, data_dir=self.data_dir
        )
        loader = DataLoader(
            train_ds, batch_size=batch_size,
            num_workers=num_workers, shuffle=False
        )

        features, _ = self._extract_features(loader, subsample=True)
        self.memory_bank = features.astype(np.float32)
        log.info(f"  Coreset size: {self.memory_bank.shape}")

        # Build FAISS index for fast k-NN search
        dim = self.memory_bank.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.memory_bank)

        # Calibrate threshold on training images.
        # Uses the same predict_image() aggregation (p99 of patch scores),
        # so the threshold is always consistent with inference.
        log.info("  Calibrating threshold on training images …")
        per_image_scores: List[float] = []
        calib_loader = DataLoader(
            train_ds, batch_size=1, num_workers=0, shuffle=False
        )
        for img_tensor, _, _ in tqdm(calib_loader, desc="  Calibrating"):
            score, _ = self.predict_image(img_tensor)
            per_image_scores.append(score)

        pct = CFG["evaluation"]["threshold_percentile"]
        self.threshold = float(np.percentile(per_image_scores, pct))
        log.info(f"  Threshold (p{pct} of {len(per_image_scores)} images): "
                 f"{self.threshold:.4f}")

        self.save()

    # -------------------------------------------------------------- #
    #  Inference                                                       #
    # -------------------------------------------------------------- #
    def _score_features(self, features: np.ndarray) -> np.ndarray:
        """
        For each patch feature vector, compute its distance to the
        nearest neighbour in the memory bank.
        """
        features = features.astype(np.float32)
        distances, _ = self.index.search(features, 1)
        # Use Euclidean L2 distance (square root of FAISS IndexFlatL2 distance)
        scores = np.sqrt(np.maximum(distances[:, 0], 0.0))
        return scores

    def predict_image(
        self,
        img_tensor: torch.Tensor,
    ) -> Tuple[float, np.ndarray]:
        """
        Parameters
        ----------
        img_tensor : (1, 3, H, W) normalised tensor

        Returns
        -------
        anomaly_score : scalar — p99 of patch scores (robust to noisy patches)
        score_map     : (H_orig, W_orig) float32 heatmap
        """
        assert self.index is not None, "Model not fitted. Call fit() or load()."

        with torch.no_grad():
            feat_map = self._get_patch_features(img_tensor.to(DEVICE))

        B, C, H, W = feat_map.shape
        patches = feat_map.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()

        # Find 1st-nearest neighbor for base PatchCore scoring
        distances, _ = self.index.search(
            patches.astype(np.float32), 1
        )
        # Convert FAISS squared L2 distance to Euclidean distance
        patch_scores = np.sqrt(np.maximum(distances[:, 0], 0.0))

        # Reshape to spatial map and upsample to input resolution
        score_map_small = patch_scores.reshape(H, W)
        score_map = self._upsample_score_map(score_map_small)

        # p99 of patch scores: robust against isolated noisy patches
        # while still catching genuine defects (which affect many patches).
        anomaly_score = float(np.percentile(patch_scores, 99))
        return anomaly_score, score_map

    @staticmethod
    def _upsample_score_map(
        score_map: np.ndarray, size: int = 224
    ) -> np.ndarray:
        """Bilinear upsample score_map (H, W) → (size, size)."""
        from PIL import Image as PILImage
        img = PILImage.fromarray(score_map.astype(np.float32))
        img = img.resize((size, size), PILImage.Resampling.BILINEAR)
        return np.array(img)

    # -------------------------------------------------------------- #
    #  Persistence                                                     #
    # -------------------------------------------------------------- #
    def save(self) -> None:
        path = self.output_dir / "patchcore.pkl"
        payload = {
            "memory_bank": self.memory_bank,
            "threshold":   self.threshold,
            "category":    self.category,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        faiss.write_index(self.index, str(self.output_dir / "faiss.index"))
        log.info(f"  Model saved to {self.output_dir}")

    def load(self) -> None:
        pkl_path   = self.output_dir / "patchcore.pkl"
        index_path = self.output_dir / "faiss.index"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"No saved model at {pkl_path}. Run train.py first."
            )
        with open(pkl_path, "rb") as f:
            payload = pickle.load(f)
        self.memory_bank = payload["memory_bank"]
        self.threshold   = payload["threshold"]
        self.index       = faiss.read_index(str(index_path))
        log.info(f"[{self.category}] Loaded model from {self.output_dir}")


# ================================================================== #
#  Convenience factory                                                 #
# ================================================================== #
def build_patchcore(category: str) -> PatchCore:
    cfg = CFG
    return PatchCore(
        category               = category,
        patch_size             = cfg["model"]["patch_size"],
        num_neighbors          = cfg["model"]["num_neighbors"],
        coreset_sampling_ratio = cfg["model"]["coreset_sampling_ratio"],
        layers                 = cfg["model"]["layers"],
        image_size             = cfg["dataset"]["image_size"],
        data_dir               = cfg["dataset"]["data_dir"],
        output_dir             = cfg["training"]["output_dir"],
    )