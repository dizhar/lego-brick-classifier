"""Tier 2 fine-grained LEGO classifier — EfficientNet-B0.

Wraps the trained Phase F model so the Streamlit app can classify individual
LEGO pieces detected by Tier 1 (YOLO11m → 43 broad categories).

The preprocessing chain matches Phase E exactly:
  - Crop bbox with 10% padding (clipped to image bounds)
  - Square letterbox with neutral gray (128,128,128) fill
  - Resize to 224×224 with LANCZOS
  - ImageNet normalization (mean/std from training)

Usage:

    from src.tier2 import load_tier2

    tier2 = load_tier2('models/efficientnet_b0_tier2_v1/best.pt', device='cpu')

    # YOLO gives you bbox_xyxy (x1, y1, x2, y2) in original image pixel coords
    preds = tier2.classify_detection(original_pil_image, bbox_xyxy, top_k=5)
    for p in preds:
        print(f"{p.rank}. bricklink {p.bricklink_identifier} "
              f"({p.broad_category_name}) — {p.confidence:.1%}")
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, cast

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier2Prediction:
    """One top-K prediction from the fine-grained classifier."""
    rank: int                       # 1-indexed (1 = best)
    class_id_dense: int             # 0..N-1 model output index
    bricklink_identifier: str       # the LEGO part ID, e.g. "3001"
    broad_category_id: int          # 0..42 — should match Tier 1's class if pipeline agrees
    broad_category_name: str        # e.g. "Bricks", "Plates and Tiles"
    confidence: float               # softmax probability, 0..1


# ---------------------------------------------------------------------------
# Preprocessing constants — must match Phase E exactly
# ---------------------------------------------------------------------------

_TARGET_SIZE = 224
_PADDING_RATIO = 0.10
_LETTERBOX_FILL = (128, 128, 128)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Pillow >= 9.1 moved resampling constants under Image.Resampling; older
# versions kept them as top-level Image.LANCZOS aliases (now deprecated).
# Resolve once at import so the type checker sees a clean attribute.
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover — Pillow < 9.1 fallback
    _LANCZOS = Image.LANCZOS  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class Tier2Classifier:
    """EfficientNet-B0 fine-grained classifier wrapper.

    The Phase F checkpoint contains everything needed for inference:
      - state_dict
      - num_classes (1607)
      - dense_to_bricklink     mapping dense_id → "3001" etc.
      - dense_to_broad         mapping dense_id → 0..42
      - sparse_to_dense / dense_to_sparse  (for compatibility with Phase E manifest)

    So we don't need the side-car class_mapping.json — best.pt is self-contained.
    """

    def __init__(self, weights_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self._load(weights_path)
        # Image tensor transform (letterboxing/cropping happens in numpy before this)
        self._tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, weights_path: Path) -> None:
        ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
        num_classes = int(ckpt.get("num_classes", 0))
        if num_classes == 0:
            # Fallback: infer from state_dict
            head_weight = ckpt["state_dict"].get("classifier.1.weight")
            if head_weight is None:
                raise RuntimeError("Can't determine num_classes from checkpoint")
            num_classes = head_weight.shape[0]

        # Build the model architecture exactly as in Phase F Cell 8
        model = efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        model.to(self.device)
        self.model = model
        self.num_classes = num_classes

        # Mappings — keys may come back as ints or str depending on torch save format
        self.dense_to_bricklink = {int(k): str(v) for k, v in ckpt["dense_to_bricklink"].items()}
        self.dense_to_broad = {int(k): int(v) for k, v in ckpt["dense_to_broad"].items()}

        # Broad-id → name comes from data.yaml (43 categories from Phase B export id=3).
        # Hard-coded here so the module is self-contained — these MUST match data.yaml
        # used by Tier 1 training, otherwise the agreement check is meaningless.
        self.broad_id_to_name = {
            0: "Bars, Ladders and Fences", 1: "Baseplates", 2: "Bricks",
            3: "Bricks Angled", 4: "Bricks Round and Cones", 5: "Bricks Wedged",
            6: "Containers", 7: "Duplo, Quatro and Primo", 8: "Electronics",
            9: "Energy Effects", 10: "Flags, Signs, Plastics and Cloth",
            11: "Hinges, Arms and Turntables", 12: "Large Buildable Figures",
            13: "Minifig Accessories", 14: "Minifig Headwear", 15: "Minifig Lower Body",
            16: "Minifig Upper Body", 17: "Minifigs", 18: "Panels and Frames",
            19: "Plants and Animals", 20: "Plates Angled", 21: "Plates Special",
            22: "Plates and Tiles", 23: "Projectiles / Launchers", 24: "Rock",
            25: "Round and Curved", 26: "Supports, Girders and Cranes",
            27: "Technic Axles", 28: "Technic Beams", 29: "Technic Beams Special",
            30: "Technic Bricks", 31: "Technic Connectors", 32: "Technic Gears",
            33: "Technic Panels", 34: "Technic Pins", 35: "Technic Special",
            36: "Technic Steering, Suspension and Engine", 37: "Tools",
            38: "Transportation - Land", 39: "Transportation - Sea and Air",
            40: "Tubes and Hoses", 41: "Wheels and Tyres",
            42: "Windscreens and Fuselage",
        }

    # ------------------------------------------------------------------
    # Crop preprocessing (matches Phase E)
    # ------------------------------------------------------------------

    @staticmethod
    def crop_with_letterbox(
        image: Image.Image,
        bbox_xyxy: Tuple[float, float, float, float],
        padding_ratio: float = _PADDING_RATIO,
        target_size: int = _TARGET_SIZE,
        fill_color: Tuple[int, int, int] = _LETTERBOX_FILL,
    ) -> Image.Image:
        """Take an original PIL image + xyxy bbox in pixel coords → 224×224 letterboxed crop."""
        x1, y1, x2, y2 = bbox_xyxy
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        pad_x = bw * padding_ratio
        pad_y = bh * padding_ratio
        img_w, img_h = image.size
        x1c = max(0, int(x1 - pad_x))
        y1c = max(0, int(y1 - pad_y))
        x2c = min(img_w, int(x2 + pad_x))
        y2c = min(img_h, int(y2 + pad_y))
        if x2c <= x1c or y2c <= y1c:
            # Degenerate bbox — return a gray square so caller doesn't crash
            return Image.new("RGB", (target_size, target_size), fill_color)

        crop = image.crop((x1c, y1c, x2c, y2c)).convert("RGB")
        w, h = crop.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), fill_color)
        canvas.paste(crop, ((side - w) // 2, (side - h) // 2))
        return canvas.resize((target_size, target_size), _LANCZOS)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def classify_crop(self, crop_224: Image.Image, top_k: int = 5) -> List[Tier2Prediction]:
        """Classify an already-preprocessed 224×224 RGB crop."""
        # cast: torchvision.transforms.Compose is typed too loosely (Any-ish) for
        # Pylance to know that ToTensor() + Normalize() actually returns a Tensor.
        tensor = cast(torch.Tensor, self._tensor_transform(crop_224))
        x = tensor.unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = logits.softmax(dim=1)[0]
        top_k = min(top_k, self.num_classes)
        topk_probs, topk_idx = probs.topk(top_k)
        results: List[Tier2Prediction] = []
        for rank, (p, idx) in enumerate(zip(topk_probs.cpu().tolist(), topk_idx.cpu().tolist())):
            broad_id = self.dense_to_broad.get(idx, -1)
            results.append(Tier2Prediction(
                rank=rank + 1,
                class_id_dense=idx,
                bricklink_identifier=self.dense_to_bricklink.get(idx, "?"),
                broad_category_id=broad_id,
                broad_category_name=self.broad_id_to_name.get(broad_id, "?"),
                confidence=float(p),
            ))
        return results

    def classify_detection(
        self,
        original_image: Image.Image,
        bbox_xyxy: Tuple[float, float, float, float],
        top_k: int = 5,
    ) -> List[Tier2Prediction]:
        """Take original image + YOLO bbox → preprocess crop → return top-K predictions."""
        crop = self.crop_with_letterbox(original_image, bbox_xyxy)
        return self.classify_crop(crop, top_k=top_k)

    @torch.no_grad()
    def classify_detections_batch(
        self,
        original_image: Image.Image,
        bboxes_xyxy: Sequence[Tuple[float, float, float, float]],
        top_k: int = 5,
    ) -> List[List[Tier2Prediction]]:
        """Batched inference for many detections from one image. Faster than calling
        classify_detection in a loop for >4 detections."""
        if not bboxes_xyxy:
            return []
        crops = [self.crop_with_letterbox(original_image, bb) for bb in bboxes_xyxy]
        # cast: see classify_crop — Pylance can't narrow Compose's return to Tensor.
        tensors: List[torch.Tensor] = [
            cast(torch.Tensor, self._tensor_transform(c)) for c in crops
        ]
        batch = torch.stack(tensors).to(self.device)
        logits = self.model(batch)
        probs = logits.softmax(dim=1)
        top_k = min(top_k, self.num_classes)
        topk_probs, topk_idx = probs.topk(top_k, dim=1)

        out: List[List[Tier2Prediction]] = []
        for det_idx in range(len(crops)):
            row: List[Tier2Prediction] = []
            for rank in range(top_k):
                idx = int(topk_idx[det_idx, rank].item())
                p = float(topk_probs[det_idx, rank].item())
                broad_id = self.dense_to_broad.get(idx, -1)
                row.append(Tier2Prediction(
                    rank=rank + 1,
                    class_id_dense=idx,
                    bricklink_identifier=self.dense_to_bricklink.get(idx, "?"),
                    broad_category_id=broad_id,
                    broad_category_name=self.broad_id_to_name.get(broad_id, "?"),
                    confidence=p,
                ))
            out.append(row)
        return out


# ---------------------------------------------------------------------------
# Streamlit-cached loader
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading Tier 2 classifier (EfficientNet-B0)...")
def load_tier2(weights_path: str, device: str = "cpu") -> Tier2Classifier:
    """Cache the model across Streamlit reruns. Pass paths as strings (cache_resource
    keys on the args — Path objects can be inconsistent here)."""
    return Tier2Classifier(Path(weights_path), device=device)


# ---------------------------------------------------------------------------
# Convenience: agreement check between Tier 1 and Tier 2
# ---------------------------------------------------------------------------


def tier1_tier2_agreement(yolo_broad_id: int, tier2_top1: Tier2Prediction) -> bool:
    """True if YOLO's broad category matches Tier 2's top-1 broad category.
    In production this is the gate for "do we trust the fine-grained prediction"."""
    return yolo_broad_id == tier2_top1.broad_category_id