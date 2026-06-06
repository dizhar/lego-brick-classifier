"""
BriqVision real-world preprocessing for YOLO inference.

Implements the pixel-value filters from the preprocessing plan, in order:

    1. CLAHE             -- local contrast enhancement (L channel only)
    2. Unsharp mask      -- edge sharpening
    3. Exposure          -- multiplicative brightness in EV stops
    4. Saturation        -- flat multiplier on HSV S channel
       OR
       LEGO hue boost    -- selective per-hue-band saturation
                            (Python port of LegoHueBoostKernel.metal)

All filters are deterministic: same input + same params always yields the same
output. No geometric changes (no crop/resize/rotate), so YOLO bbox coordinates
on the preprocessed image align pixel-for-pixel with the original.

This Python implementation mirrors what the iOS app will do with Core Image /
Metal: same filter order, same parameters, same intent. Use it to A/B test
the preprocessing chain in Streamlit before porting to Swift.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessParams:
    """Conservative defaults from the BriqVision preprocessing plan."""
    # CLAHE -- local contrast enhancement
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    # Unsharp mask -- edge sharpening
    unsharp_radius: float = 2.0
    unsharp_intensity: float = 0.5
    # Exposure normalization (+0.3 EV ~= 1.23x brightness)
    exposure_ev: float = 0.3
    # Saturation gain on HSV S channel (1.10 = +10%)
    saturation_gain: float = 1.10


def apply_clahe(img_rgb: np.ndarray, clip_limit: float = 2.0,
                tile_grid_size: int = 8) -> np.ndarray:
    """
    Local contrast enhancement on the L channel of LAB color space.

    Applying CLAHE to L only preserves color; applying it per-RGB-channel
    causes color shifts. Tile size controls how local the equalization is;
    smaller tiles = more aggressive local contrast.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)


def apply_unsharp(img_rgb: np.ndarray, radius: float = 2.0,
                  intensity: float = 0.5) -> np.ndarray:
    """
    Unsharp mask: out = img + intensity * (img - blur(img)).

    Subtracting a slightly-blurred copy from the original amplifies edge
    transitions. Too aggressive creates bright halos that look like false
    boundaries to the model.
    """
    blurred = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=radius)
    return cv2.addWeighted(img_rgb, 1.0 + intensity, blurred, -intensity, 0)


def apply_exposure(img_rgb: np.ndarray, ev: float = 0.3) -> np.ndarray:
    """
    Exposure adjustment in EV stops. +1 EV doubles brightness; +0.3 EV ~= 1.23x.

    Most phone cameras slightly underexpose to protect highlights, so a small
    positive bump tends to help recover dark pieces.
    """
    gain = 2.0 ** ev
    return np.clip(img_rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def apply_saturation(img_rgb: np.ndarray, gain: float = 1.10) -> np.ndarray:
    """
    Multiply HSV S channel. Pushes muddy lighting-mixed colors apart.

    Conservative gains (1.05-1.15) are safe; large gains push colors outside
    the training distribution.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# -----------------------------------------------------------------------------
# LEGO hue boost -- Python port of LegoHueBoostKernel.metal
# -----------------------------------------------------------------------------

@dataclass
class HueBoostParams:
    """
    Selective hue-band saturation control. Mirrors the Metal kernel:

      - boost_strength    : saturation multiplier in LEGO-color hue bands (>= 1.0)
      - mute_strength     : saturation multiplier in background hue bands (<= 1.0)
      - transition_width  : soft band edge width in degrees

    Hue centers and half-widths define which hues count as "LEGO" vs
    "background". These are my Python defaults derived from common LEGO color
    palettes -- tune them to match the .metal shader if it uses different bands.
    """
    boost_strength: float = 1.6
    mute_strength: float = 0.35
    transition_width: float = 8.0  # degrees

    # LEGO-characteristic hues (degrees, 0-360): saturated primary/secondary
    # colors found across the LEGO catalog. Each entry is a band CENTER.
    boost_hue_centers: Tuple[float, ...] = (0.0, 60.0, 120.0, 200.0, 240.0, 300.0)
    boost_band_half_width: float = 20.0  # degrees on each side of center

    # Background hues to mute: warm browns/tans common to wood, cardboard,
    # carpet, skin tones, packaging.
    mute_hue_centers: Tuple[float, ...] = (25.0,)
    mute_band_half_width: float = 15.0


def _build_hue_gain_lut(params: HueBoostParams) -> np.ndarray:
    """
    Precompute a 360-entry hue→saturation-gain lookup table.

    Building the LUT once per frame is ~360 cheap iterations (microseconds);
    applying it is then a single indexed multiply per pixel (sub-millisecond
    even on 4K input). The Metal kernel does the equivalent math per-pixel
    on the GPU; we trade per-pixel branching for a precomputed table.
    """
    lut = np.ones(360, dtype=np.float32)
    for h in range(360):
        gain = 1.0

        # Boost bands -- take the max gain across overlapping bands
        for center in params.boost_hue_centers:
            d = min(abs(h - center), 360 - abs(h - center))
            edge = params.boost_band_half_width + params.transition_width
            if d <= edge:
                # Ramp: 0 at outer edge, 1 inside band core
                ramp = min(1.0, max(0.0, (edge - d) / params.transition_width))
                gain = max(gain, 1.0 + (params.boost_strength - 1.0) * ramp)

        # Mute bands -- take the min gain (mute wins over boost on overlap)
        for center in params.mute_hue_centers:
            d = min(abs(h - center), 360 - abs(h - center))
            edge = params.mute_band_half_width + params.transition_width
            if d <= edge:
                ramp = min(1.0, max(0.0, (edge - d) / params.transition_width))
                gain = min(gain, 1.0 - (1.0 - params.mute_strength) * ramp)

        lut[h] = gain
    return lut


def apply_hue_boost(img_rgb: np.ndarray,
                    params: Optional[HueBoostParams] = None) -> np.ndarray:
    """
    Selectively boost LEGO-characteristic hues and mute background hues.

    Python equivalent of LegoHueBoostKernel.metal -- same parameter semantics
    so tuning that works in Streamlit ports directly to the iOS Metal kernel.
    """
    if params is None:
        params = HueBoostParams()

    lut = _build_hue_gain_lut(params)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    # OpenCV stores hue as 0-179 (half degrees); double to get 0-358 then mod 360
    h_deg = (hsv[..., 0].astype(np.int32) * 2) % 360
    gain = lut[h_deg]

    s_new = np.clip(hsv[..., 1].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    hsv[..., 1] = s_new
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def preprocess(img_rgb: np.ndarray,
               params: Optional[PreprocessParams] = None,
               hue_boost: Optional[HueBoostParams] = None) -> np.ndarray:
    """
    Apply the full filter chain in plan order.

    Args:
        img_rgb:    H x W x 3 uint8 RGB array
        params:     PreprocessParams (uses defaults if None)
        hue_boost:  if provided, REPLACES the flat saturation step with a
                    selective LEGO-aware hue boost. Both manipulate the HSV
                    S channel; running them stacked is double-counting.

    Returns:
        H x W x 3 uint8 RGB array, same dimensions as input.
    """
    if params is None:
        params = PreprocessParams()
    out = apply_clahe(img_rgb, params.clahe_clip_limit, params.clahe_tile_grid_size)
    out = apply_unsharp(out, params.unsharp_radius, params.unsharp_intensity)
    out = apply_exposure(out, params.exposure_ev)
    if hue_boost is not None:
        out = apply_hue_boost(out, hue_boost)
    else:
        out = apply_saturation(out, params.saturation_gain)
    return out