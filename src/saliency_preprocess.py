"""
BriqVision Saliency-Guided Preprocessing
=========================================
Adds a saliency mask step that isolates LEGO pieces from the background
before applying enhancement filters. Solves the core problem shown by
Grad-CAM: the model looks at fabric/surface texture instead of pieces.

Usage in Streamlit:
    from saliency_preprocess import SaliencyPreprocessor
    
    sp = SaliencyPreprocessor()
    result = sp.preprocess(image_bgr, params)
    
    # result.display_image  → raw image for user display
    # result.model_image    → preprocessed image for YOLO inference
    # result.saliency_mask  → the mask (for debug visualization)
    # result.edges          → detected edges (for debug visualization)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PreprocessResult:
    """Container for preprocessing outputs."""
    display_image: np.ndarray       # Raw image (what user sees)
    model_image: np.ndarray         # Preprocessed image (what model sees)
    saliency_mask: np.ndarray       # Binary/soft mask of detected objects
    edges: np.ndarray               # Edge detection output (debug)
    piece_regions: int              # Number of piece-like regions found
    mask_coverage: float            # % of image covered by saliency mask


@dataclass
class PreprocessParams:
    """All tunable parameters, mappable to Streamlit sliders."""
    # --- Saliency mask ---
    saliency_method: str = "edges"  # "edges", "grabcut", "bilateral"
    canny_low: int = 50             # Canny edge lower threshold
    canny_high: int = 150           # Canny edge upper threshold
    edge_dilate_px: int = 15        # Dilate edges to form regions (px)
    min_contour_area: int = 500     # Minimum contour area (pixels²)
    mask_blur_px: int = 21          # Gaussian blur on mask for soft edges
    background_suppression: float = 0.3  # How much to darken background (0=black, 1=no change)

    # --- Enhancement filters (applied to salient regions) ---
    clahe_clip: float = 2.0         # CLAHE clip limit
    clahe_tile: int = 8             # CLAHE tile grid size
    sharpen_radius: float = 2.0     # Unsharp mask radius (px)
    sharpen_intensity: float = 0.5  # Unsharp mask strength
    exposure_ev: float = 0.3        # Exposure adjustment (EV)
    saturation_gain: float = 1.1    # Saturation multiplier


def detect_edges_saliency(image_bgr: np.ndarray, params: PreprocessParams) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Detect piece-like objects using edge detection + contour analysis.
    
    Returns:
        mask: soft saliency mask (0-1 float), same size as input
        edges: raw Canny edge output (for debug display)
        n_regions: number of piece-like contours found
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce fabric/surface texture noise before edge detection.
    # LEGO piece edges are sharp and geometric; fabric texture is soft and random.
    # The blur eliminates fabric grain while preserving piece boundaries.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection — finds strong gradients (piece edges)
    edges = cv2.Canny(blurred, params.canny_low, params.canny_high)
    
    # Dilate edges to connect nearby edge fragments into solid regions.
    # A single LEGO brick edge is thin; dilation fills the interior.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (params.edge_dilate_px, params.edge_dilate_px)
    )
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Close small gaps between edge regions
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours — each closed region is a candidate piece
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter: keep only contours above minimum area (removes noise dots)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= params.min_contour_area]
    
    # Draw filled contours as mask
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, valid_contours, -1, color=(255,), thickness=cv2.FILLED)
    
    # Soft edges: blur the mask so the transition between piece and background
    # is gradual, avoiding hard cutoff artifacts that confuse the model
    blur_size = params.mask_blur_px
    if blur_size % 2 == 0:
        blur_size += 1  # Must be odd
    mask_soft = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask_float = mask_soft.astype(np.float32) / 255.0
    
    return mask_float, edges, len(valid_contours)


def detect_bilateral_saliency(image_bgr: np.ndarray, params: PreprocessParams) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Alternative saliency using bilateral filter + adaptive threshold.
    Better for images where pieces have uniform color against textured backgrounds.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter: smooths texture while preserving edges
    # This is the key difference — it specifically attacks fabric/surface texture
    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Adaptive threshold: handles uneven lighting
    binary = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=31, C=-5
    )
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.edge_dilate_px, params.edge_dilate_px))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= params.min_contour_area]
    
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, valid_contours, -1, color=(255,), thickness=cv2.FILLED)
    
    blur_size = params.mask_blur_px if params.mask_blur_px % 2 == 1 else params.mask_blur_px + 1
    mask_soft = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask_float = mask_soft.astype(np.float32) / 255.0
    
    edges = cv2.Canny(filtered, 50, 150)  # For debug display
    return mask_float, edges, len(valid_contours)


def apply_enhancement_filters(image_bgr: np.ndarray, params: PreprocessParams) -> np.ndarray:
    """
    Apply the 4-stage enhancement chain.
    Same filters as the plan, but applied AFTER saliency masking.
    """
    result = image_bgr.copy()
    
    # 1. CLAHE — local contrast enhancement
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip,
        tileGridSize=(params.clahe_tile, params.clahe_tile)
    )
    lab[:, :, 0] = clahe.apply(l_channel)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # 2. Unsharp mask — edge sharpening
    if params.sharpen_intensity > 0:
        blurred = cv2.GaussianBlur(result, (0, 0), params.sharpen_radius)
        result = cv2.addWeighted(
            result, 1.0 + params.sharpen_intensity,
            blurred, -params.sharpen_intensity,
            0
        )
    
    # 3. Exposure adjustment
    if abs(params.exposure_ev) > 0.01:
        # EV adjustment: multiply by 2^EV
        factor = np.power(2.0, params.exposure_ev)
        result = np.clip(result.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    # 4. Saturation boost
    if abs(params.saturation_gain - 1.0) > 0.01:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * params.saturation_gain, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return result


class SaliencyPreprocessor:
    """
    Main preprocessor class.
    
    Pipeline:
    1. Detect salient regions (LEGO pieces) via edge detection
    2. Create soft mask separating pieces from background
    3. Suppress background (reduce texture that distracts the model)
    4. Enhance piece regions (CLAHE + sharpen + exposure + saturation)
    5. Composite: enhanced pieces on suppressed background
    """
    
    def preprocess(self, image_bgr: np.ndarray, params: PreprocessParams) -> PreprocessResult:
        """
        Full preprocessing pipeline.
        
        Args:
            image_bgr: input image in BGR format (OpenCV default)
            params: all tunable parameters
            
        Returns:
            PreprocessResult with display image, model image, and debug outputs
        """
        h, w = image_bgr.shape[:2]
        
        # Step 1: Detect salient regions
        if params.saliency_method == "bilateral":
            mask, edges, n_regions = detect_bilateral_saliency(image_bgr, params)
        else:
            mask, edges, n_regions = detect_edges_saliency(image_bgr, params)
        
        # Step 2: Apply enhancement filters to the full image
        enhanced = apply_enhancement_filters(image_bgr, params)
        
        # Step 3: Composite — blend enhanced pieces with suppressed background
        # Expand mask to 3 channels
        mask_3ch = np.stack([mask] * 3, axis=-1)
        
        # Background: darkened/muted version of original (suppress texture)
        background = (image_bgr.astype(np.float32) * params.background_suppression).astype(np.uint8)
        
        # Composite: where mask is 1.0 → enhanced piece, where mask is 0.0 → suppressed background
        model_image = (
            enhanced.astype(np.float32) * mask_3ch +
            background.astype(np.float32) * (1.0 - mask_3ch)
        ).astype(np.uint8)
        
        # Calculate mask coverage
        mask_coverage = float(np.mean(mask > 0.5)) * 100.0
        
        return PreprocessResult(
            display_image=image_bgr,
            model_image=model_image,
            saliency_mask=mask,
            edges=edges,
            piece_regions=n_regions,
            mask_coverage=mask_coverage,
        )


# ============================================================
# STREAMLIT INTEGRATION
# ============================================================

def add_saliency_sidebar(st) -> PreprocessParams:
    """
    Add all preprocessing sliders to the Streamlit sidebar.
    Call this in your app and pass the returned params to preprocess().
    
    Usage:
        params = add_saliency_sidebar(st)
        result = SaliencyPreprocessor().preprocess(image, params)
    """
    params = PreprocessParams()
    
    with st.sidebar:
        st.subheader("🎯 Saliency mask")
        
        params.saliency_method = st.selectbox(
            "Detection method",
            ["edges", "bilateral"],
            help="'edges' uses Canny edge detection. "
                 "'bilateral' uses bilateral filter + adaptive threshold — "
                 "better for uniform pieces on textured backgrounds."
        )
        
        params.canny_low = st.slider(
            "Canny low threshold", 10, 200, 50,
            help="Lower = more edges detected. Decrease if pieces are missed."
        )
        params.canny_high = st.slider(
            "Canny high threshold", 50, 300, 150,
            help="Upper edge threshold. Should be 2-3x the low threshold."
        )
        params.edge_dilate_px = st.slider(
            "Edge dilation (px)", 3, 40, 15,
            help="How much to expand detected edges into filled regions. "
                 "Increase for larger pieces, decrease for small pieces."
        )
        params.min_contour_area = st.slider(
            "Min contour area (px²)", 100, 5000, 500,
            help="Ignore detected regions smaller than this. "
                 "Filters out noise and fabric texture artifacts."
        )
        params.mask_blur_px = st.slider(
            "Mask blur (px)", 3, 51, 21, step=2,
            help="Softens mask edges. Higher = smoother transition "
                 "between piece and background."
        )
        params.background_suppression = st.slider(
            "Background suppression", 0.0, 1.0, 0.3,
            help="0.0 = black background (strongest). "
                 "1.0 = no suppression. "
                 "0.3 = 70% darker background (recommended start)."
        )
        
        st.subheader("🔧 Enhancement filters")
        
        params.clahe_clip = st.slider("CLAHE clip limit", 0.5, 8.0, 2.0)
        params.clahe_tile = st.slider("CLAHE tile size", 4, 16, 8)
        params.sharpen_radius = st.slider("Sharpen radius (px)", 0.5, 5.0, 2.0)
        params.sharpen_intensity = st.slider("Sharpen intensity", 0.0, 1.5, 0.5)
        params.exposure_ev = st.slider("Exposure (EV)", -1.0, 1.0, 0.3)
        params.saturation_gain = st.slider("Saturation gain", 0.5, 1.5, 1.1)
    
    return params


def show_debug_panel(st, result: PreprocessResult):
    """
    Display debug visualizations in the Streamlit main area.
    Shows the saliency mask, edge detection, and before/after comparison.
    """
    st.subheader("🎯 Saliency Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(
            cv2.cvtColor(result.display_image, cv2.COLOR_BGR2RGB),
            caption="Original (display path)",
            use_container_width=True,
        )
    
    with col2:
        # Colorize the saliency mask for visualization
        mask_vis = (result.saliency_mask * 255).astype(np.uint8)
        mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
        st.image(
            cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB),
            caption=f"Saliency mask ({result.piece_regions} regions, {result.mask_coverage:.1f}% coverage)",
            use_container_width=True,
        )
    
    with col3:
        st.image(
            cv2.cvtColor(result.model_image, cv2.COLOR_BGR2RGB),
            caption="Model input (enhanced pieces, suppressed background)",
            use_container_width=True,
        )
    
    # Edge detection debug
    with st.expander("Edge detection detail"):
        st.image(result.edges, caption="Canny edges", use_container_width=True)


# ============================================================
# EXAMPLE: Minimal Streamlit app
# ============================================================
#
# import streamlit as st
# import cv2
# from saliency_preprocess import SaliencyPreprocessor, add_saliency_sidebar, show_debug_panel
#
# st.title("BriqVision Preprocessing Lab")
#
# uploaded = st.file_uploader("Upload image", type=["jpg", "png"])
# if uploaded:
#     file_bytes = np.frombuffer(uploaded.read(), np.uint8)
#     image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#
#     params = add_saliency_sidebar(st)
#     sp = SaliencyPreprocessor()
#     result = sp.preprocess(image, params)
#
#     show_debug_panel(st, result)
#
#     # Feed result.model_image to YOLO...
#     model = YOLO("best.pt")
#     detections = model(cv2.cvtColor(result.model_image, cv2.COLOR_BGR2RGB))
#     ...