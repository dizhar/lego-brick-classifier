import streamlit as st
from PIL import Image
import numpy as np
from ultralytics.models.yolo import YOLO
import cv2
import colorsys

from .utils import load_class_names
from .config import (
    MODEL_PATH, 
    CLASS_NAMES_PATH, 
    PAGE_TITLE, 
    PAGE_ICON,
    TOP_K_PREDICTIONS,
    EXAMPLE_IMAGES
)
from .session_state import (
    init_session_state,
    set_example_image,
    set_uploaded_image,
    get_selected_example,
    is_example_selected
)

# Page config
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide"
)

def get_color_for_class(class_id, num_classes=80):
    """Generate a distinct color for each class using HSV color space"""
    # Use golden ratio for better color distribution
    golden_ratio = 0.618033988749895
    hue = (class_id * golden_ratio) % 1.0
    # Use high saturation and value for vibrant colors
    rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
    # Convert to 0-255 range and return as BGR for OpenCV
    return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

def draw_text_with_outline(img, text, position, font, font_scale, text_color, outline_color, thickness, outline_thickness):
    """Draw text with an outline for better visibility"""
    x, y = position
    # Draw outline
    cv2.putText(img, text, (x, y), font, font_scale, outline_color, thickness + outline_thickness, cv2.LINE_AA)
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

@st.cache_data
def load_classes():
    """Load class names (cached)"""
    return load_class_names(CLASS_NAMES_PATH)

@st.cache_resource
def load_model():
    """Load YOLO model (cached)"""
    model = YOLO(MODEL_PATH)
    return model

def process_image(image, model, class_names, confidence_threshold, iou_threshold, 
                  show_labels, show_confidence):
    """
    Process an image and return detection results.
    
    Args:
        image: PIL Image
        model: YOLO model
        class_names: Dictionary of class names
        confidence_threshold: Confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        show_labels: Whether to show labels
        show_confidence: Whether to show confidence scores
        
    Returns:
        tuple: (annotated_image, num_detections, detection_data)
    """
    # Make prediction with filters
    with st.spinner('Detecting LEGO pieces...'):
        results = model(
            image, 
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False
        )
        result = results[0]
    
    # Get detection information
    num_detections = len(result.boxes)
    
    if num_detections > 0:
        boxes = result.boxes
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        
        # Get unique detected piece types
        unique_classes = np.unique(classes)
        detected_piece_names = [class_names.get(int(c), f"Class {c}") 
                               for c in unique_classes]
        
        detection_data = {
            'confidences': confidences,
            'classes': classes,
            'xyxy': xyxy,
            'detected_piece_names': detected_piece_names,
            'result': result
        }
        
        return image, num_detections, detection_data
    else:
        return image, 0, None

def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Initialize selected detection index in session state
    if 'selected_detection_idx' not in st.session_state:
        st.session_state.selected_detection_idx = None
    
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.write("Upload an image to detect and classify LEGO pieces")
    
    # Load model and class names
    try:
        model = load_model()
        class_names = load_classes()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure your model file and class_names.json are in the correct locations.")
        st.stop()
    
    # Sidebar for controls
    st.sidebar.header("Detection Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Filter detections below this confidence level"
    )
    
    iou_threshold = st.sidebar.slider(
        "IoU Threshold (NMS)",
        min_value=0.0,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Non-Maximum Suppression threshold to reduce overlapping boxes"
    )
    
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)
    
    # ===== NEW: Example Images Section =====
    st.markdown("---")
    st.subheader("🚀 Quick Start - Try an Example")
    st.write("Click on an example image below to see instant detection results:")
    
    # Create 3 columns for example images
    col1, col2, col3 = st.columns(3)
    
    example_keys = ['simple', 'complex', 'mixed']
    columns = [col1, col2, col3]
    
    for col, example_key in zip(columns, example_keys):
        with col:
            example_info = EXAMPLE_IMAGES[example_key]
            example_path = example_info['path']
            
            # Display thumbnail
            try:
                if example_path.exists():
                    thumbnail = Image.open(example_path)
                    st.image(thumbnail, use_column_width=True)
                    
                    # Button to select this example
                    button_label = f"📷 {example_info['label']}"
                    button_type = "primary" if is_example_selected(example_key) else "secondary"
                    
                    if st.button(button_label, key=f"btn_{example_key}", type=button_type):
                        set_example_image(example_key)
                        st.rerun()
                    
                    # Description
                    st.caption(example_info['description'])
                else:
                    st.warning(f"Example image not found: {example_path}")
            except Exception as e:
                st.error(f"Error loading example: {e}")
    
    st.markdown("---")
    # ===== END: Example Images Section =====
    
    # File uploader
    st.subheader("📤 Or Upload Your Own Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    # Determine which image to process
    image_to_process = None
    
    # Priority: uploaded file overrides example
    if uploaded_file is not None:
        set_uploaded_image()
        image_to_process = Image.open(uploaded_file).convert('RGB')
    elif get_selected_example() is not None:
        # Load example image
        example_key = get_selected_example()
        example_path = EXAMPLE_IMAGES[example_key]['path']
        try:
            image_to_process = Image.open(example_path).convert('RGB')
        except Exception as e:
            st.error(f"Error loading example image: {e}")
            st.stop()
    
    # Process image if we have one
    if image_to_process is not None:
        # Process the image
        image, num_detections, detection_data = process_image(
            image_to_process, model, class_names, 
            confidence_threshold, iou_threshold,
            show_labels, show_confidence
        )
        
        if num_detections > 0 and detection_data is not None:
            confidences = detection_data['confidences']
            classes = detection_data['classes']
            xyxy = detection_data['xyxy']
            detected_piece_names = detection_data['detected_piece_names']
            result = detection_data['result']
            
            # Add filter by piece type in sidebar
            st.sidebar.markdown("---")
            st.sidebar.subheader("Filter by Piece Type")
            selected_pieces = st.sidebar.multiselect(
                "Select piece types to display:",
                options=detected_piece_names,
                default=detected_piece_names,
                help="Choose which LEGO piece types to show"
            )
            
            # Filter detections based on selected pieces
            if selected_pieces:
                # Create reverse mapping: name -> class_id
                name_to_id = {name: idx for idx, name in class_names.items()}
                
                # Get indices of selected classes
                selected_class_ids = [name_to_id[name] for name in selected_pieces if name in name_to_id]
                
                # Create mask for selected classes
                mask = np.isin(classes, selected_class_ids)
                
                # Filter all arrays
                filtered_confidences = confidences[mask]
                filtered_classes = classes[mask]
                filtered_xyxy = xyxy[mask]
                
                num_filtered = len(filtered_confidences)
            else:
                # If no pieces selected, show nothing
                filtered_confidences = np.array([])
                filtered_classes = np.array([])
                filtered_xyxy = np.array([])
                num_filtered = 0
            
            # Display annotated image with detections
            st.subheader("Detected Pieces")
            
            # Create a filtered result for visualization
            if num_filtered > 0:
                # Always use custom drawing with adaptive font scaling
                annotated_image = np.array(image)
                
                # Determine which detections to draw
                # If a specific detection is selected, only draw that one
                if st.session_state.selected_detection_idx is not None:
                    indices_to_draw = [st.session_state.selected_detection_idx]
                else:
                    # Draw all filtered detections
                    indices_to_draw = range(num_filtered)
                
                # Calculate adaptive font scale based on number of detections
                # Fewer pieces = larger boxes = need smaller font
                # More pieces = smaller boxes = can use larger font
                if num_filtered <= 3:
                    font_scale = 0.25  # Smaller for 1-3 pieces
                elif num_filtered <= 10:
                    font_scale = 0.3   # Medium-small for 4-10 pieces
                elif num_filtered <= 30:
                    font_scale = 0.35  # Medium for 11-30 pieces
                else:
                    font_scale = 0.4   # Default for many pieces
                
                # Draw filtered boxes with distinct colors
                for i in indices_to_draw:
                    if i >= num_filtered:  # Safety check
                        continue
                        
                    box = filtered_xyxy[i].astype(int)
                    conf = filtered_confidences[i]
                    cls = filtered_classes[i]
                    
                    # Get class name
                    cls_int = int(cls)
                    class_name = class_names.get(cls_int, f"Class {cls_int}")
                    
                    # Get color for this class
                    color = get_color_for_class(cls_int)
                    
                    # Draw box with thicker line
                    cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), color, 3)
                    
                    # Draw label
                    if show_labels:
                        label = f"{class_name}"
                        if show_confidence:
                            label += f" {conf:.2f}"
                        
                        # Get text size - adaptive font scale
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        thickness = 1
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )
                        
                        # Draw background rectangle for label - reduced padding
                        padding = 2
                        label_y = max(box[1] - 3, text_height + padding)
                        cv2.rectangle(
                            annotated_image,
                            (box[0], label_y - text_height - padding),
                            (box[0] + text_width + padding * 2, label_y + padding),
                            color,
                            -1
                        )
                        
                        # Draw text with black color and white outline for maximum readability
                        draw_text_with_outline(
                            annotated_image,
                            label,
                            (box[0] + padding, label_y),
                            font,
                            font_scale,
                            (0, 0, 0),  # Black text
                            (255, 255, 255),  # White outline
                            thickness,
                            1  # Outline thickness
                        )
                
                # Convert BGR to RGB for display
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    
                st.image(annotated_image_rgb, use_column_width=True)
            else:
                st.image(image, use_column_width=True)
                st.warning("No pieces match the selected filters.")
            
            # Display detection results
            if st.session_state.selected_detection_idx is not None:
                st.info(f"Showing individual piece #{st.session_state.selected_detection_idx + 1} of {num_filtered} detected piece(s)")
                # Add button to show all pieces again
                if st.button("🔄 Show All Pieces", type="primary"):
                    st.session_state.selected_detection_idx = None
                    st.rerun()
            else:
                st.success(f"Detection Complete! Showing {num_filtered} of {num_detections} detected piece(s)")
            
            if num_filtered > 0:
                # Sort by confidence (highest first)
                sorted_indices = np.argsort(filtered_confidences)[::-1]
                
                st.write("### Detected LEGO Pieces:")
                st.caption("Click 'Show Only This' to highlight individual pieces on the image")
                
                # Create a nice table view
                for i, idx in enumerate(sorted_indices[:20]):  # Show top 20
                    class_id = filtered_classes[idx]
                    confidence = filtered_confidences[idx]
                    box = filtered_xyxy[idx]
                    
                    # Get class name
                    class_id_int = int(class_id)
                    class_name = class_names.get(class_id_int, f"Class {class_id_int}")
                    
                    # Calculate box size for reference
                    box_width = box[2] - box[0]
                    box_height = box[3] - box[1]
                    
                    # Determine if this piece is currently selected
                    is_selected = (st.session_state.selected_detection_idx == idx)
                    
                    with st.expander(f"#{i+1}: {class_name} ({confidence*100:.1f}%)", expanded=(i<5 or is_selected)):
                        col_a, col_b, col_c = st.columns([2, 2, 3])
                        with col_a:
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                        with col_b:
                            st.metric("Box Size", f"{int(box_width)}x{int(box_height)}px")
                        with col_c:
                            # Button to show only this piece
                            button_label = "✅ Selected" if is_selected else "👁️ Show Only This"
                            button_type = "secondary" if is_selected else "primary"
                            
                            if st.button(button_label, key=f"show_piece_{idx}", type=button_type):
                                if is_selected:
                                    # Deselect if already selected
                                    st.session_state.selected_detection_idx = None
                                else:
                                    # Select this piece
                                    st.session_state.selected_detection_idx = idx
                                st.rerun()
                
                # Show detection statistics in sidebar
                st.sidebar.markdown("---")
                st.sidebar.subheader("Detection Statistics")
                st.sidebar.metric("Total Detections", num_detections)
                st.sidebar.metric("Filtered Detections", num_filtered)
                st.sidebar.metric("Avg Confidence", f"{filtered_confidences.mean()*100:.1f}%")
                
                # Count detections by class
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