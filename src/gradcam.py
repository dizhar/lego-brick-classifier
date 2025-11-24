"""
GradCAM / EigenCAM explainability for YOLOv8/YOLOv11 detection models.

Provides the YOLOGradCAM wrapper plus analysis helpers that turn the raw
activation map into actionable metrics: per-detection activation, image-level
focus ratio, and missed-piece candidates (hot CAM peaks outside any box).
"""

import cv2
import numpy as np
import torch

from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

_GRADIENT_FREE = {"eigencam": EigenCAM}
_GRADIENT_BASED = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus}

_INPUT_SIZE = 640  # YOLO default inference resolution


class YOLOGradCAM:
    """Class Activation Map wrapper for an Ultralytics YOLO model."""

    def __init__(self, model, method="eigencam", target_layer_index=None):
        self.method = method.lower()
        self.torch_model = model.model
        self.torch_model.eval()
        self.device = next(self.torch_model.parameters()).device

        self.target_layer = self._resolve_target_layer(target_layer_index)

        if self.method in _GRADIENT_FREE:
            cam_cls = _GRADIENT_FREE[self.method]
        elif self.method in _GRADIENT_BASED:
            cam_cls = _GRADIENT_BASED[self.method]
        else:
            raise ValueError(
                f"Unknown CAM method '{method}'. "
                f"Choose from: {list(_GRADIENT_FREE) + list(_GRADIENT_BASED)}"
            )

        self.cam = cam_cls(model=self.torch_model, target_layers=[self.target_layer])

    def _resolve_target_layer(self, target_layer_index):
        layers = list(self.torch_model.model.children())
        if target_layer_index is not None:
            return layers[target_layer_index]
        for layer in reversed(layers):
            if layer.__class__.__name__ == "Detect":
                continue
            if len(list(layer.parameters())) > 0:
                return layer
        raise ValueError("Could not auto-resolve a target layer; pass target_layer_index.")

    def generate(self, image_rgb: np.ndarray, return_raw: bool = False):
        """
        Produce a heatmap for an RGB image.

        Args:
            image_rgb:   H x W x 3 uint8 RGB array
            return_raw:  if True, returns (overlay, raw_cam) where raw_cam is
                         a H x W float32 array in [0, 1]

        Returns:
            overlay (uint8 H x W x 3 RGB) — or (overlay, raw_cam) if return_raw
        """
        h, w = image_rgb.shape[:2]

        img_resized = cv2.resize(image_rgb, (_INPUT_SIZE, _INPUT_SIZE))
        img_float = img_resized.astype(np.float32) / 255.0
        input_tensor = (
            torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self.device)
        )

        # Dummy target -- EigenCAM ignores it but BaseCAM's None-handling tries
        # to argmax YOLO's tuple output and crashes.
        dummy_targets = [ClassifierOutputTarget(0)]
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=dummy_targets)[0]  # type: ignore[arg-type]

        overlay = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        overlay_resized = cv2.resize(overlay, (w, h))

        if return_raw:
            raw_cam_resized = cv2.resize(grayscale_cam, (w, h))
            return overlay_resized, raw_cam_resized
        return overlay_resized


# -----------------------------------------------------------------------------
# Analysis helpers -- turn a raw CAM + detection boxes into actionable metrics.
# -----------------------------------------------------------------------------

def _clip_box(box, h, w):
    """Clip xyxy to image bounds, return ints. Returns None if degenerate."""
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def analyze_detections(cam: np.ndarray, boxes_xyxy: np.ndarray,
                       hot_threshold: float = 0.5) -> list:
    """
    Per-detection CAM statistics.

    Args:
        cam:           H x W float array in [0, 1]
        boxes_xyxy:    N x 4 array of [x1, y1, x2, y2]
        hot_threshold: pixel-value cutoff for "hot pixel" coverage metric

    Returns:
        list of N dicts with keys: mean, max, hot_coverage
        (hot_coverage = fraction of pixels in box above hot_threshold)
    """
    h, w = cam.shape
    out = []
    for box in boxes_xyxy:
        clipped = _clip_box(box, h, w)
        if clipped is None:
            out.append({"mean": 0.0, "max": 0.0, "hot_coverage": 0.0})
            continue
        x1, y1, x2, y2 = clipped
        crop = cam[y1:y2, x1:x2]
        out.append({
            "mean": float(crop.mean()),
            "max": float(crop.max()),
            "hot_coverage": float((crop > hot_threshold).mean()),
        })
    return out


def image_stats(cam: np.ndarray, boxes_xyxy: np.ndarray) -> dict:
    """Image-level CAM stats: is attention focused on detections or background?"""
    h, w = cam.shape
    mask = np.zeros(cam.shape, dtype=bool)
    for box in boxes_xyxy:
        clipped = _clip_box(box, h, w)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        mask[y1:y2, x1:x2] = True

    cam_in = cam[mask] if mask.any() else np.array([0.0])
    cam_out = cam[~mask] if (~mask).any() else np.array([0.0])

    mean_in = float(cam_in.mean())
    mean_out = float(cam_out.mean())
    total = float(cam.sum())
    inside_sum = float(cam[mask].sum()) if mask.any() else 0.0

    return {
        "mean_inside": mean_in,
        "mean_outside": mean_out,
        "focus_ratio": mean_in / max(mean_out, 1e-6),
        "in_box_heat_fraction": inside_sum / max(total, 1e-6),
        "box_area_fraction": float(mask.mean()),
    }


def find_missed_candidates(cam: np.ndarray, boxes_xyxy: np.ndarray,
                           top_k: int = 5, threshold: float = 0.6,
                           min_distance_px: int = 30) -> list:
    """
    Hot peaks NOT inside any detection box -- potential false negatives.

    Uses simple greedy non-max suppression: pick the highest pixel outside
    all boxes, suppress everything within `min_distance_px`, repeat.
    """
    h, w = cam.shape
    masked = cam.copy()
    for box in boxes_xyxy:
        clipped = _clip_box(box, h, w)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        masked[y1:y2, x1:x2] = 0.0

    candidates = []
    while len(candidates) < top_k:
        idx = int(masked.argmax())
        y, x = np.unravel_index(idx, masked.shape)
        val = float(masked[y, x])
        if val < threshold:
            break
        candidates.append({"x": int(x), "y": int(y), "value": val})
        y0, y1_ = max(0, y - min_distance_px), min(h, y + min_distance_px)
        x0, x1_ = max(0, x - min_distance_px), min(w, x + min_distance_px)
        masked[y0:y1_, x0:x1_] = 0.0

    return candidates