import streamlit as st
from PIL import Image
import numpy as np
from ultralytics.models.yolo import YOLO
import cv2
import colorsys


from .gradcam import (
    YOLOGradCAM,
    analyze_detections,
    image_stats,
    find_missed_candidates,
)
from .preprocess import preprocess, PreprocessParams
from .tier2 import load_tier2, tier1_tier2_agreement
from .db import load_part_catalog, format_part_label
from .utils import load_class_names
from .config import (
    MODEL_PATH,
    CLASS_NAMES_PATH,
    PAGE_TITLE,
    PAGE_ICON,
    TOP_K_PREDICTIONS,
    EXAMPLE_IMAGES,
    TIER2_WEIGHTS_PATH,
    TIER2_DEFAULT_DEVICE,
    TIER2_TOP_K,
)
from .session_state import (
    init_session_state,
    set_example_image,
    set_uploaded_image,
    get_selected_example,
    is_example_selected
)

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide"
)


def get_color_for_class(class_id, num_classes=80):
    """Generate a distinct color for each class using HSV color space"""
    golden_ratio = 0.618033988749895
    hue = (class_id * golden_ratio) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))


def draw_text_with_outline(img, text, position, font, font_scale, text_color, outline_color, thickness, outline_thickness):
    """Draw text with an outline for better visibility"""
    x, y = position
    cv2.putText(img, text, (x, y), font, font_scale, outline_color, thickness + outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


@st.cache_data
def load_classes():
    return load_class_names(CLASS_NAMES_PATH)


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


@st.cache_resource
def load_gradcam(_model):
    return YOLOGradCAM(_model, method="eigencam", target_layer_index=19)


def process_image(image, model, class_names, confidence_threshold, iou_threshold,
                  show_labels, show_confidence):
    """Run YOLO on `image` and return (image, num_detections, detection_data)."""
    with st.spinner('Detecting LEGO pieces...'):
        results = model(image, conf=confidence_threshold, iou=iou_threshold, verbose=False)
        result = results[0]

    num_detections = len(result.boxes)

    if num_detections > 0:
        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        unique_classes = np.unique(classes)
        detected_piece_names = [class_names.get(int(c), f"Class {c}") for c in unique_classes]
        detection_data = {
            'confidences': confidences,
            'classes': classes,
            'xyxy': xyxy,
            'detected_piece_names': detected_piece_names,
            'result': result,
        }
        return image, num_detections, detection_data
    return image, 0, None


def build_preprocess_params(clahe_clip, clahe_tile, unsharp_r, unsharp_i,
                            exposure_ev, sat_gain):
    return PreprocessParams(
        clahe_clip_limit=clahe_clip,
        clahe_tile_grid_size=clahe_tile,
        unsharp_radius=unsharp_r,
        unsharp_intensity=unsharp_i,
        exposure_ev=exposure_ev,
        saturation_gain=sat_gain,
    )


def main():
    init_session_state()

    if 'selected_detection_idx' not in st.session_state:
        st.session_state.selected_detection_idx = None

    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.write("Upload an image to detect and classify LEGO pieces")

    try:
        model = load_model()
        class_names = load_classes()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure your model file and class_names.json are in the correct locations.")
        st.stop()

    # ===== Sidebar: detection controls =====
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
        help="Filter detections below this confidence level"
    )
    iou_threshold = st.sidebar.slider(
        "IoU Threshold (NMS)", 0.0, 1.0, 0.45, 0.05,
        help="Non-Maximum Suppression threshold to reduce overlapping boxes"
    )
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)

    # ===== Sidebar: preprocessing =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Preprocessing")
    enable_preprocess = st.sidebar.checkbox(
        "Enable preprocessing",
        value=False,
        help="Apply CLAHE → sharpen → exposure → saturation before YOLO. "
             "The model sees the enhanced image; you see the original."
    )
    show_preprocessed = st.sidebar.checkbox(
        "Show what the model sees",
        value=False,
        disabled=not enable_preprocess,
        help="Side-by-side comparison: original vs preprocessed input."
    )
    with st.sidebar.expander("Preprocessing parameters", expanded=False):
        clahe_clip = st.slider("CLAHE clip limit", 0.5, 8.0, 2.0, 0.5,
                               disabled=not enable_preprocess,
                               help="Higher = more aggressive local contrast.")
        clahe_tile = st.slider("CLAHE tile size", 4, 16, 8, 2,
                               disabled=not enable_preprocess,
                               help="Smaller = finer-grained local contrast.")
        unsharp_r = st.slider("Sharpen radius (px)", 0.5, 5.0, 2.0, 0.5,
                              disabled=not enable_preprocess)
        unsharp_i = st.slider("Sharpen intensity", 0.0, 1.5, 0.5, 0.1,
                              disabled=not enable_preprocess,
                              help="Too high creates bright edge halos.")
        exposure_ev = st.slider("Exposure (EV)", -1.0, 1.0, 0.3, 0.1,
                                disabled=not enable_preprocess)
        sat_gain = st.slider("Saturation gain", 0.5, 1.5, 1.10, 0.05,
                             disabled=not enable_preprocess)

    # ===== Sidebar: explainability =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Explainability")
    show_gradcam = st.sidebar.checkbox(
        "Show Activation Heatmap (EigenCAM)",
        value=False,
        help="Activation map + per-detection analytics."
    )

    # ===== Sidebar: Tier 2 fine-grained classification =====
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔎 Tier 2 (fine-grained)")
    use_tier2 = st.sidebar.checkbox(
        "Classify each detection with EfficientNet-B0",
        value=False,
        help="After YOLO finds pieces, run Tier 2 (EfficientNet-B0) on each crop to "
             "predict the specific bricklink part ID (1,607 classes)."
    )
    tier2_top_k = st.sidebar.slider(
        "Tier 2 top-K", 1, 5, TIER2_TOP_K,
        disabled=not use_tier2,
        help="How many candidate part IDs to show per detection."
    )

    # Load the Tier 2 model only if requested (cached across reruns)
    tier2 = None
    part_catalog: dict = {}
    if use_tier2:
        try:
            tier2 = load_tier2(str(TIER2_WEIGHTS_PATH), device=TIER2_DEFAULT_DEVICE)
        except FileNotFoundError:
            st.sidebar.error(
                f"Tier 2 weights not found at `{TIER2_WEIGHTS_PATH}`. "
                "Download with:\n\n"
                "```\ngcloud storage cp "
                "gs://briqvision-training/training_weights/efficientnet_b0_tier2_v1/best.pt "
                f"{TIER2_WEIGHTS_PATH}\n```"
            )
            use_tier2 = False
        except Exception as e:
            st.sidebar.error(f"Could not load Tier 2 model: {e}")
            use_tier2 = False

        # Try to load the part catalog from the DB — labels become part names
        # instead of bricklink IDs. Graceful fallback if DB unavailable.
        if use_tier2:
            try:
                part_catalog = load_part_catalog()
                st.sidebar.caption(f"📚 Part catalog: {len(part_catalog)} parts loaded")
            except Exception as e:
                st.sidebar.warning(
                    f"Could not load part catalog from DB ({type(e).__name__}). "
                    "Labels will show bricklink IDs instead of part names."
                )
                with st.sidebar.expander("DB error details", expanded=False):
                    st.code(str(e))

    # ===== Example images =====
    st.markdown("---")
    st.subheader("🚀 Quick Start - Try an Example")
    st.write("Click on an example image below to see instant detection results:")

    col1, col2, col3 = st.columns(3)
    example_keys = ['simple', 'complex', 'mixed']
    columns = [col1, col2, col3]

    for col, example_key in zip(columns, example_keys):
        with col:
            example_info = EXAMPLE_IMAGES[example_key]
            example_path = example_info['path']
            try:
                if example_path.exists():
                    thumbnail = Image.open(example_path)
                    st.image(thumbnail, use_column_width=True)
                    button_label = f"📷 {example_info['label']}"
                    button_type = "primary" if is_example_selected(example_key) else "secondary"
                    if st.button(button_label, key=f"btn_{example_key}", type=button_type):
                        set_example_image(example_key)
                        st.rerun()
                    st.caption(example_info['description'])
                else:
                    st.warning(f"Example image not found: {example_path}")
            except Exception as e:
                st.error(f"Error loading example: {e}")

    st.markdown("---")

    st.subheader("📤 Or Upload Your Own Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    image_to_process = None
    if uploaded_file is not None:
        set_uploaded_image()
        image_to_process = Image.open(uploaded_file).convert('RGB')
    elif get_selected_example() is not None:
        example_key = get_selected_example()
        example_path = EXAMPLE_IMAGES[example_key]['path']
        try:
            image_to_process = Image.open(example_path).convert('RGB')
        except Exception as e:
            st.error(f"Error loading example image: {e}")
            st.stop()

    if image_to_process is not None:
        # --- What the MODEL sees vs what the USER sees ---
        original_np = np.array(image_to_process)
        if enable_preprocess:
            with st.spinner("Applying preprocessing…"):
                params = build_preprocess_params(
                    clahe_clip, clahe_tile, unsharp_r, unsharp_i, exposure_ev, sat_gain
                )
                preprocessed_np = preprocess(original_np, params)
                model_input = Image.fromarray(preprocessed_np)
        else:
            preprocessed_np = original_np
            model_input = image_to_process

        if enable_preprocess and show_preprocessed:
            st.subheader("👁️ What the model sees")
            st.caption(
                "Bounding boxes below are drawn on the original (left), but YOLO "
                "is processing the enhanced version (right). Pixel coordinates "
                "align because preprocessing only changes pixel values, not geometry."
            )
            ppcol1, ppcol2 = st.columns(2)
            with ppcol1:
                st.image(image_to_process, caption="Original (displayed)",
                         use_column_width=True)
            with ppcol2:
                st.image(preprocessed_np, caption="Preprocessed (model input)",
                         use_column_width=True)
            st.markdown("---")

        # Inference on (possibly) preprocessed image
        _, num_detections, detection_data = process_image(
            model_input, model, class_names,
            confidence_threshold, iou_threshold,
            show_labels, show_confidence
        )
        # All downstream display uses the ORIGINAL image -- coords align because
        # preprocessing applied only pixel-value transforms.
        image = image_to_process

        # ===== Activation heatmap + analytics =====
        if show_gradcam:
            st.subheader("🔥 Activation Heatmap & Analytics")
            if enable_preprocess:
                st.caption(
                    "Heatmap reflects activations on the **preprocessed** image — "
                    "compare these metrics against a run with preprocessing off "
                    "to validate filter effectiveness."
                )
            try:
                gradcam = load_gradcam(model)
                cam_input = preprocessed_np if enable_preprocess else original_np
                with st.spinner("Generating activation heatmap…"):
                    heatmap, raw_cam = gradcam.generate(cam_input, return_raw=True)

                hcol1, hcol2 = st.columns(2)
                with hcol1:
                    st.image(image, caption="Original", use_column_width=True)
                with hcol2:
                    st.image(heatmap, caption="EigenCAM Overlay", use_column_width=True)

                if detection_data is not None:
                    boxes = detection_data['xyxy']
                    cls_arr = detection_data['classes']
                    conf_arr = detection_data['confidences']

                    stats = image_stats(raw_cam, boxes)
                    per_det = analyze_detections(raw_cam, boxes)
                    missed = find_missed_candidates(raw_cam, boxes, top_k=5)

                    st.markdown("#### 📊 Image-level focus")
                    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                    mcol1.metric("Focus ratio", f"{stats['focus_ratio']:.2f}×",
                                 help="Mean CAM inside boxes ÷ outside. >1 is good.")
                    mcol2.metric("Heat in boxes", f"{stats['in_box_heat_fraction']*100:.0f}%")
                    mcol3.metric("Mean CAM inside", f"{stats['mean_inside']:.3f}")
                    mcol4.metric("Mean CAM outside", f"{stats['mean_outside']:.3f}")

                    st.markdown("#### 🎯 Per-detection activation")
                    st.caption("Low mean CAM + high confidence = brittle detection.")
                    rows = []
                    for i, (det_stats, c, cf) in enumerate(zip(per_det, cls_arr, conf_arr)):
                        rows.append({
                            "#": i + 1,
                            "Class": class_names.get(int(c), f"Class {int(c)}"),
                            "Confidence": f"{cf*100:.1f}%",
                            "Mean CAM": round(det_stats["mean"], 3),
                            "Max CAM": round(det_stats["max"], 3),
                            "Hot coverage": f"{det_stats['hot_coverage']*100:.0f}%",
                        })
                    st.dataframe(rows, use_container_width=True, hide_index=True)

                    st.markdown("#### 🔍 Potential missed pieces")
                    if missed:
                        st.caption(
                            f"Found {len(missed)} hot region(s) outside any detection box."
                        )
                        miss_rows = [
                            {"#": i + 1, "x": m["x"], "y": m["y"],
                             "Peak CAM": round(m["value"], 3)}
                            for i, m in enumerate(missed)
                        ]
                        st.dataframe(miss_rows, use_container_width=True, hide_index=True)
                    else:
                        st.success("No strong activation peaks outside detections. ✓")
                else:
                    st.info("No detections — lower the confidence threshold for analytics.")
            except Exception as e:
                st.error(f"Could not generate heatmap: {e}")
            st.markdown("---")

        if num_detections > 0 and detection_data is not None:
            confidences = detection_data['confidences']
            classes = detection_data['classes']
            xyxy = detection_data['xyxy']
            detected_piece_names = detection_data['detected_piece_names']
            result = detection_data['result']

            st.sidebar.markdown("---")
            st.sidebar.subheader("Filter by Piece Type")
            selected_pieces = st.sidebar.multiselect(
                "Select piece types to display:",
                options=detected_piece_names,
                default=detected_piece_names,
                help="Choose which LEGO piece types to show"
            )

            if selected_pieces:
                name_to_id = {name: idx for idx, name in class_names.items()}
                selected_class_ids = [name_to_id[name] for name in selected_pieces if name in name_to_id]
                mask = np.isin(classes, selected_class_ids)
                filtered_confidences = confidences[mask]
                filtered_classes = classes[mask]
                filtered_xyxy = xyxy[mask]
                num_filtered = len(filtered_confidences)
            else:
                filtered_confidences = np.array([])
                filtered_classes = np.array([])
                filtered_xyxy = np.array([])
                num_filtered = 0

            st.subheader("Detected Pieces")

            # --- Tier 2 fine-grained classification (batched) ---
            # Run on ALL filtered detections so we can put part IDs on every box.
            # Result is indexed by filtered detection position (0..num_filtered-1).
            # We pass image_to_process (the ORIGINAL image), not the preprocessed
            # one — Tier 2 was trained on un-preprocessed synthetic data. The bbox
            # geometry is identical either way (preprocessing is geometry-preserving).
            tier2_results = []
            if use_tier2 and tier2 is not None and num_filtered > 0:
                all_bboxes = [tuple(filtered_xyxy[i].astype(float)) for i in range(num_filtered)]
                with st.spinner(f"Running Tier 2 on {num_filtered} detection(s)…"):
                    tier2_results = tier2.classify_detections_batch(
                        image_to_process, all_bboxes, top_k=max(tier2_top_k, 1)
                    )

            # Pre-compute the disagreement mask (Tier 1 broad cat ≠ Tier 2 broad cat).
            # When Tier 2 is on and we're in multi-piece mode, hide these boxes —
            # they're the least trustworthy detections.
            disagreement_mask = [False] * num_filtered
            if use_tier2 and tier2_results:
                for i in range(num_filtered):
                    if i < len(tier2_results) and tier2_results[i]:
                        if tier2_results[i][0].broad_category_id != int(filtered_classes[i]):
                            disagreement_mask[i] = True
            hidden_disagreement_count = sum(disagreement_mask)

            # In single-piece "Show Only This" mode we always draw the chosen box,
            # even on disagreement — the user explicitly asked to see it.
            showing_single = st.session_state.selected_detection_idx is not None

            if num_filtered > 0:
                annotated_image = np.array(image)
                indices_to_draw = ([st.session_state.selected_detection_idx]
                                   if showing_single
                                   else range(num_filtered))
                if num_filtered <= 3:
                    font_scale = 0.25
                elif num_filtered <= 10:
                    font_scale = 0.3
                elif num_filtered <= 30:
                    font_scale = 0.35
                else:
                    font_scale = 0.4

                drawn_count = 0
                for i in indices_to_draw:
                    # Type-narrow for Pylance: indices_to_draw can contain a value
                    # sourced from st.session_state (typed Any|None). i is never
                    # None here at runtime, but the static checker can't tell.
                    if i is None or i >= num_filtered:
                        continue

                    # Skip disagreement boxes in multi-piece mode (user asked for this).
                    if (not showing_single) and disagreement_mask[i]:
                        continue

                    box = filtered_xyxy[i].astype(int)
                    conf = filtered_confidences[i]
                    cls = filtered_classes[i]
                    cls_int = int(cls)
                    class_name = class_names.get(cls_int, f"Class {cls_int}")
                    color = get_color_for_class(cls_int)
                    cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), color, 3)
                    drawn_count += 1

                    # Tier 2 top-1 for THIS detection (if Tier 2 is enabled)
                    t2_pred = tier2_results[i][0] if (use_tier2 and i < len(tier2_results) and tier2_results[i]) else None

                    if show_labels:
                        # When Tier 2 is on, its part NAME is the label (looked up
                        # from the DB catalog). When Tier 2 is off, fall back to
                        # the original YOLO label.
                        if t2_pred is not None:
                            # NOTE: cv2 Hershey font is ASCII only, so we can't use
                            # ⚠ here — it renders as ???. Use "!" as a disagreement
                            # marker (Tier 2's predicted broad cat ≠ YOLO's broad cat).
                            disagree_marker = "" if t2_pred.broad_category_id == cls_int else " !"
                            # Look up the human-readable part name; falls back to
                            # "#<bricklink_id>" if the catalog isn't loaded or the
                            # part isn't found.
                            part_label = format_part_label(part_catalog, t2_pred.bricklink_identifier)
                            label = f"{part_label} {t2_pred.confidence:.2f}{disagree_marker}"
                        else:
                            label = f"{class_name}"
                            if show_confidence:
                                label += f" {conf:.2f}"

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        thickness = 1
                        padding = 2

                        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        block_height = text_height + padding * 2

                        # Place label block above the box; clip to image top if needed.
                        block_top = box[1] - block_height - 3
                        if block_top < 0:
                            block_top = box[1] + 3  # not enough room above → put inside top of box

                        # Filled background rectangle behind the text
                        cv2.rectangle(
                            annotated_image,
                            (box[0], block_top),
                            (box[0] + text_width + padding * 2, block_top + block_height),
                            color, -1
                        )

                        # Text
                        text_y = block_top + text_height + padding
                        draw_text_with_outline(
                            annotated_image, label,
                            (box[0] + padding, text_y),
                            font, font_scale,
                            (0, 0, 0), (255, 255, 255),
                            thickness, 1
                        )

                st.image(annotated_image, use_column_width=True)
            else:
                st.image(image, use_column_width=True)
                st.warning("No pieces match the selected filters.")

            if st.session_state.selected_detection_idx is not None:
                st.info(f"Showing individual piece #{st.session_state.selected_detection_idx + 1} of {num_filtered} detected piece(s)")
                if st.button("🔄 Show All Pieces", type="primary"):
                    st.session_state.selected_detection_idx = None
                    st.rerun()
            else:
                if use_tier2 and hidden_disagreement_count > 0:
                    kept = num_filtered - hidden_disagreement_count
                    st.success(
                        f"Detection Complete! Showing **{kept} of {num_filtered}** trusted detection(s). "
                        f"Hidden **{hidden_disagreement_count}** where Tier 1 and Tier 2 disagree on the broad category."
                    )
                else:
                    st.success(f"Detection Complete! Showing {num_filtered} of {num_detections} detected piece(s)")

            if num_filtered > 0:
                sorted_indices = np.argsort(filtered_confidences)[::-1]

                st.write("### Detected LEGO Pieces:")
                st.caption("Click 'Show Only This' to highlight individual pieces on the image")

                for i, idx in enumerate(sorted_indices[:TOP_K_PREDICTIONS]):
                    class_id = filtered_classes[idx]
                    confidence = filtered_confidences[idx]
                    box = filtered_xyxy[idx]
                    class_id_int = int(class_id)
                    class_name = class_names.get(class_id_int, f"Class {class_id_int}")
                    box_width = box[2] - box[0]
                    box_height = box[3] - box[1]
                    is_selected = (st.session_state.selected_detection_idx == idx)

                    with st.expander(f"#{i+1}: {class_name} ({confidence*100:.1f}%)", expanded=(i<5 or is_selected)):
                        col_a, col_b, col_c = st.columns([2, 2, 3])
                        with col_a:
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                        with col_b:
                            st.metric("Box Size", f"{int(box_width)}x{int(box_height)}px")
                        with col_c:
                            button_label = "✅ Selected" if is_selected else "👁️ Show Only This"
                            button_type = "secondary" if is_selected else "primary"
                            if st.button(button_label, key=f"show_piece_{idx}", type=button_type):
                                if is_selected:
                                    st.session_state.selected_detection_idx = None
                                else:
                                    st.session_state.selected_detection_idx = idx
                                st.rerun()

                        # --- Tier 2 predictions for this detection ---
                        # tier2_results is indexed by filtered detection index (idx),
                        # not by position in sorted_indices (i).
                        if use_tier2 and idx < len(tier2_results) and tier2_results[idx]:
                            preds = tier2_results[idx]
                            top1 = preds[0]
                            agree = tier1_tier2_agreement(class_id_int, top1)
                            badge = "✅" if agree else "⚠️"
                            st.markdown(
                                f"{badge} **Tier 2 top-1:** part `{top1.bricklink_identifier}` "
                                f"({top1.broad_category_name}) — {top1.confidence*100:.1f}%"
                            )
                            if not agree:
                                st.caption(
                                    f"YOLO broad category (**{class_name}**) disagrees with "
                                    f"Tier 2 top-1 broad category (**{top1.broad_category_name}**). "
                                    "Either Tier 1 was wrong about the broad category, or Tier 2 "
                                    "is confused — inspect lower-ranked predictions below."
                                )

                            # Show full top-K as a small table
                            pred_rows = []
                            for p in preds:
                                pred_rows.append({
                                    "#": p.rank,
                                    "Part (BrickLink)": p.bricklink_identifier,
                                    "Broad category": p.broad_category_name,
                                    "Same as Tier 1?": "✓" if p.broad_category_id == class_id_int else "✗",
                                    "Confidence": f"{p.confidence*100:.1f}%",
                                })
                            st.dataframe(pred_rows, use_container_width=True, hide_index=True)

                st.sidebar.markdown("---")
                st.sidebar.subheader("Detection Statistics")
                st.sidebar.metric("Total Detections", num_detections)
                st.sidebar.metric("Filtered Detections", num_filtered)
                st.sidebar.metric("Avg Confidence", f"{filtered_confidences.mean()*100:.1f}%")

                unique_filtered, counts = np.unique(filtered_classes, return_counts=True)
                st.sidebar.write("**Filtered Pieces by Type:**")
                for class_id, count in sorted(zip(unique_filtered, counts), key=lambda x: x[1], reverse=True)[:10]:
                    class_id_int = int(class_id)
                    if class_id_int in class_names:
                        st.sidebar.write(f"• {class_names[class_id_int]}: {count}")
        else:
            st.image(image, use_column_width=True)
            st.warning("No LEGO pieces detected. Try lowering the confidence threshold in the sidebar.")


if __name__ == "__main__":
    main()