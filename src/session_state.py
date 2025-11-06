"""
Session state management for LEGO Classifier App
Handles state for example images and uploaded files
"""
import streamlit as st


def init_session_state():
    """
    Initialize session state variables if they don't exist.
    Call this at the start of the app.
    """
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = None
    
    if 'image_source' not in st.session_state:
        st.session_state.image_source = None
    
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None


def set_example_image(example_key):
    """
    Set the selected example image.
    
    Args:
        example_key (str): Key from EXAMPLE_IMAGES dict ('simple', 'complex', 'mixed')
    """
    st.session_state.selected_example = example_key
    st.session_state.image_source = 'example'


def set_uploaded_image():
    """
    Set the image source to uploaded file.
    Clears any selected example.
    """
    st.session_state.selected_example = None
    st.session_state.image_source = 'upload'
    st.session_state.current_image_path = None


def clear_image_selection():
    """
    Clear all image selections.
    Useful for reset functionality.
    """
    st.session_state.selected_example = None
    st.session_state.image_source = None
    st.session_state.current_image_path = None


def get_current_image_source():
    """
    Get the current image source type.
    
    Returns:
        str or None: 'example', 'upload', or None
    """
    return st.session_state.get('image_source', None)


def get_selected_example():
    """
    Get the currently selected example key.
    
    Returns:
        str or None: Example key ('simple', 'complex', 'mixed') or None
    """
    return st.session_state.get('selected_example', None)


def is_example_selected(example_key):
    """
    Check if a specific example is currently selected.
    
    Args:
        example_key (str): Example key to check
        
    Returns:
        bool: True if this example is selected
    """
    return (st.session_state.get('image_source') == 'example' and 
            st.session_state.get('selected_example') == example_key)