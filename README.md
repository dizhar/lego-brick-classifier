# 🧱 LEGO Classifier

A Streamlit web application for detecting and classifying LEGO pieces using a YOLO object detection model. Upload an image or pick an example scene to get real-time detections across **43 broad LEGO categories**, with optional image preprocessing and an EigenCAM explainability view to inspect what the model is actually looking at.

## ✨ Features

- **Real-time Object Detection**: Detect and classify LEGO pieces from uploaded images using a YOLO model
- **43 Broad Categories**: Classifies pieces into high-level groups (Bricks, Plates, Minifigs, Technic parts, Wheels, and more — see [`data/class_names.json`](data/class_names.json))
- **Tunable Detection**: Adjust the confidence threshold and IoU (NMS) threshold live from the sidebar
- **Advanced Visualization**:
  - Color-coded bounding boxes (distinct, stable color per class via golden-ratio HSV)
  - Optional labels and confidence scores with outlined text for readability
  - Toggle individual pieces on/off to isolate a single detection on the image
- **Image Preprocessing (optional)**: Apply a deterministic, geometry-preserving filter chain before inference — CLAHE → unsharp mask → exposure → saturation — and view a side-by-side of "what you see" vs. "what the model sees". Includes an optional LEGO-aware hue boost.
- **Explainability (EigenCAM)**: Generate an activation heatmap overlay plus analytics:
  - Image-level focus ratio (attention inside vs. outside detection boxes)
  - Per-detection activation stats (mean/max CAM, hot coverage) to spot brittle detections
  - Potential missed pieces — hot activation peaks that fall outside any detection box
- **Filter by Piece Type**: Show only selected piece categories
- **Detection Statistics**: Total/filtered counts, average confidence, and per-type breakdown in the sidebar
- **Example Image Gallery**: Try the classifier instantly with simple, complex, and mixed-type example scenes
- **Caching**: Cached model, class names, and EigenCAM resources for fast reruns

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

4. **Download the model weights**

   Weights are not stored in git — they live in Google Cloud Storage
   (`gs://briqvision-training`) and are pulled in with a script. Make sure the
   [gcloud CLI](https://cloud.google.com/sdk/docs/install) is installed and
   authenticated (`gcloud auth application-default login`), then run:

   ```bash
   ./scripts/download_models.sh
   ```

   This fetches the Tier 1 YOLO model to `models/best.pt` and the Tier 2
   EfficientNet model to `models/efficientnet_b0_tier2_v1/best.pt`. Pass
   `--force` to re-download.

5. **Verify required files**
   - `models/best.pt` (Tier 1 YOLO) and `models/efficientnet_b0_tier2_v1/best.pt` (Tier 2) exist
   - `data/class_names.json` exists
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

## 🕹️ Using the App

1. **Pick an image** — click an example scene or upload your own (`jpg`, `jpeg`, `png`).
2. **Tune detection** — adjust the **Confidence** and **IoU (NMS)** sliders in the sidebar; toggle labels and confidence scores.
3. **(Optional) Preprocess** — enable preprocessing to enhance the image before YOLO runs. Turn on *"Show what the model sees"* for a side-by-side comparison, and open *Preprocessing parameters* to tune CLAHE, sharpening, exposure, and saturation.
4. **(Optional) Explain** — enable the **Activation Heatmap (EigenCAM)** to overlay the model's attention and review focus/per-detection/missed-piece analytics.
5. **Inspect** — filter by piece type, expand individual detections, and use *"Show Only This"* to isolate a single piece on the image.

## 📁 Project Structure

```
lego-classifier-app/
├── main.py                 # Application entry point — runs the Streamlit app
├── requirements.txt        # Python dependencies
│
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── app.py              # Main Streamlit application logic and UI
│   ├── model.py            # YOLO model loading and inference
│   ├── preprocess.py       # Deterministic preprocessing chain (CLAHE, unsharp, exposure, saturation, hue boost)
│   ├── gradcam.py          # EigenCAM wrapper + activation analytics helpers
│   ├── saliency_preprocess.py # Saliency-based preprocessing experiments
│   ├── utils.py            # Helper functions (image processing, class names)
│   ├── config.py           # Configuration settings and paths
│   └── session_state.py    # Streamlit session state management
│
├── models/                 # Trained model weights
│   └── best.pt             # YOLO trained model
│
├── data/                   # Data files
│   ├── class_names.json    # LEGO piece class labels (43 broad categories)
│   └── examples/           # Example images for testing
│       ├── example_simple.png
│       ├── example_complex.png
│       └── example_mixed.png
│
├── notebooks/              # Jupyter notebooks for preprocessing & training
│   ├── Preprocessing_LEGO_Dataset_Corrected_(3) (3).ipynb
│   └── presentation_yolo_broad_category_training.ipynb
│
├── assets/                 # Static assets (logos, icons)
│
├── tests/                  # Unit tests
│   ├── test.py
│   ├── test_model.py
│   └── test_gradcam.py
│
└── .vscode/                # VS Code settings
```

## 🎨 Preprocessing

`src/preprocess.py` implements a deterministic, geometry-preserving filter chain applied before inference. Because the filters only change pixel values (no crop/resize/rotate), YOLO bounding-box coordinates on the preprocessed image align pixel-for-pixel with the original.

Chain order:

1. **CLAHE** — local contrast enhancement on the LAB L channel (preserves color)
2. **Unsharp mask** — edge sharpening
3. **Exposure** — multiplicative brightness in EV stops
4. **Saturation** — flat HSV S-channel multiplier, **or** an optional **LEGO hue boost** that selectively boosts LEGO-characteristic hues and mutes background hues (a Python port of a Metal kernel, for A/B testing before porting to iOS)

Defaults live in `PreprocessParams` / `HueBoostParams` and can all be tuned live from the sidebar.

## 🔬 Explainability (EigenCAM)

`src/gradcam.py` provides `YOLOGradCAM`, a Class Activation Map wrapper for Ultralytics YOLO models (supports `eigencam`, `gradcam`, and `gradcam++`; the app uses EigenCAM). Beyond the heatmap overlay, it turns the raw activation map into actionable metrics:

- `image_stats()` — image-level focus ratio, in-box heat fraction, and box-area fraction
- `analyze_detections()` — per-box mean/max CAM and hot-pixel coverage
- `find_missed_candidates()` — hot CAM peaks outside every detection box (potential false negatives), found via greedy non-max suppression

## 🔧 Configuration

The application can be configured via `src/config.py`:

- `MODEL_PATH`: Path to the YOLO model weights
- `CLASS_NAMES_PATH`: Path to class names JSON file
- `EXAMPLES_DIR`: Directory containing example images
- `EXAMPLE_IMAGES`: Example scene definitions (path, label, description)
- `TOP_K_PREDICTIONS`: Number of top predictions to display (default: 5)
- `PAGE_TITLE` and `PAGE_ICON`: Application branding

## 📊 Model Training

Check the `notebooks/` directory for the preprocessing and training pipelines:

1. **Preprocessing_LEGO_Dataset_Corrected_(3) (3).ipynb**
   - Dataset preprocessing and augmentation
   - Data cleaning and validation
   - Train/test split generation

2. **presentation_yolo_broad_category_training.ipynb**
   - YOLO training over the 43 broad LEGO categories
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
python tests/test_gradcam.py
python tests/test.py
```

## 📦 Dependencies

Main dependencies (see `requirements.txt` for exact versions):

- **streamlit** (1.39.0) — Web application framework
- **torch** (≥2.0.0) / **torchvision** (≥0.15.0) — PyTorch deep learning framework
- **ultralytics** (≥8.0.0) — YOLO implementation
- **grad-cam** (≥1.5.0) — EigenCAM / Grad-CAM explainability
- **opencv-python-headless** (≥4.8.0) — Image processing
- **Pillow** (10.4.0) — Image handling
- **numpy** (<2.0.0) — Numerical operations
- **pyyaml** (≥6.0) — Config/dataset parsing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- YOLO by Ultralytics
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for EigenCAM explainability
- Streamlit for the web framework
- LEGO dataset contributors
