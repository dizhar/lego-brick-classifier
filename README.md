# 🧱 LEGO Classifier

A Streamlit web application for classifying LEGO pieces using deep learning with YOLOv8 object detection. This application provides real-time detection and classification of LEGO pieces from uploaded images or example scenes.

## ✨ Features

- **Real-time Object Detection**: Upload images to detect and classify LEGO pieces using YOLOv8
- **Advanced Visualization**:
  - Color-coded bounding boxes with distinct colors for each class
  - Confidence scores displayed on each detection
  - Text with outline for better visibility
  - Side-by-side comparison of original and annotated images
- **Top K Predictions**: See the top 5 most confident predictions for each detected piece
- **Example Image Gallery**: Test the classifier with pre-loaded example images:
  - Simple scenes with few pieces
  - Complex scenes with many pieces
  - Mixed type collections
- **Interactive UI**: Clean, responsive Streamlit interface with wide layout
- **Session State Management**: Seamless navigation between uploaded and example images
- **Caching**: Optimized performance with cached model loading and class names

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster inference

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd lego-classifier-app
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv venv

   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify required files**
   - Ensure trained YOLOv8 model exists at `models/best.pt`
   - Ensure class names file exists at `data/class_names.json`
   - (Optional) Add example images to `data/examples/`

### Running the Application

Start the Streamlit app:

```bash
streamlit run main.py
```

The application will open automatically in your default browser at `http://localhost:8501`

### Development Mode

Run with auto-reload for development:

```bash
streamlit run main.py --server.runOnSave true
```

## 📁 Project Structure

```
lego-classifier-app/
├── main.py                 # Application entry point - imports and runs app
├── requirements.txt        # Python dependencies (Streamlit, PyTorch, YOLOv8)
│
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── app.py              # Main Streamlit application logic and UI
│   ├── model.py            # YOLOv8 model loading and inference
│   ├── utils.py            # Helper functions (image processing, class names)
│   ├── config.py           # Configuration settings and paths
│   └── session_state.py    # Streamlit session state management
│
├── models/                 # Trained model weights
│   └── best.pt             # YOLOv8 trained model (52MB)
│
├── data/                   # Data files
│   ├── class_names.json    # LEGO piece class labels mapping
│   └── examples/           # Example images for testing
│       ├── example_simple.png
│       ├── example_complex.png
│       └── example_mixed.png
│
├── notebooks/              # Jupyter notebooks for training
│   ├── Preprocessing_LEGO_Dataset_Corrected_(3) (3).ipynb
│   └── presentation_yolo_broad_category_training.ipynb
│
├── assets/                 # Static assets (logos, icons)
│
├── tests/                  # Unit tests
│   ├── test.py
│   └── test_model.py
│
└── .vscode/                # VS Code settings
```

## 🔧 Configuration

The application can be configured via `src/config.py`:

- `MODEL_PATH`: Path to YOLOv8 model weights
- `CLASS_NAMES_PATH`: Path to class names JSON file
- `EXAMPLES_DIR`: Directory containing example images
- `TOP_K_PREDICTIONS`: Number of top predictions to display (default: 5)
- `PAGE_TITLE` and `PAGE_ICON`: Application branding

## 📊 Model Training

Check the `notebooks/` directory for training pipelines:

### Available Notebooks:

1. **Preprocessing*LEGO_Dataset_Corrected*(3) (3).ipynb**

   - Dataset preprocessing and augmentation
   - Data cleaning and validation
   - Train/test split generation

2. **presentation_yolo_broad_category_training.ipynb**
   - YOLOv8 model training with broad category classification
   - Hyperparameter tuning
   - Performance evaluation and metrics

## 🧪 Testing

Run unit tests:

```bash
python -m pytest tests/
```

Run specific test files:

```bash
python tests/test_model.py
python tests/test.py
```

## 📦 Dependencies

Main dependencies (see `requirements.txt` for versions):

- **streamlit** (1.39.0) - Web application framework
- **torch** (≥2.0.0) - PyTorch deep learning framework
- **ultralytics** (≥8.0.0) - YOLOv8 implementation
- **opencv-python-headless** (≥4.8.0) - Image processing
- **Pillow** (10.4.0) - Image handling
- **numpy** (<2.0.0) - Numerical operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Streamlit for the web framework
- LEGO dataset contributors
