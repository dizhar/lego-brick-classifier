# 🧱 LEGO Classifier

A Streamlit web application for classifying LEGO pieces using deep learning with YOLOv8 object detection.

## Features

- **Real-time Object Detection**: Upload images to detect and classify LEGO pieces using YOLOv8
- **Bounding Box Visualization**: View detected pieces with color-coded bounding boxes and confidence scores
- **Top Predictions**: See the top K most confident predictions for each detected piece
- **Example Images**: Test the classifier with pre-loaded example images
- **Interactive UI**: Clean, responsive interface with image upload and gallery view
- **Session State Management**: Seamless navigation between uploaded and example images

## Setup

1. Clone the repository
2. Create virtual environment:

```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
   pip install -r requirements.txt
```

4. Add your trained YOLOv8 model to `models/best.pt`
5. Add your class names to `data/class_names.json`

## Run

```bash
streamlit run main.py
```

## Project Structure

```
lego-classifier-app/
├── main.py              # Entry point
├── src/                 # Source code
│   ├── app.py           # Streamlit app logic
│   ├── model.py         # Model loading and inference
│   ├── utils.py         # Helper functions
│   ├── config.py        # Configuration settings
│   └── session_state.py # Session state management
├── models/              # Trained models
│   └── best.pt          # YOLOv8 model weights
├── data/                # Data files
│   ├── class_names.json # LEGO piece class labels
│   └── examples/        # Example images for testing
├── notebooks/           # Jupyter notebooks
│   ├── Preprocessing_LEGO_Dataset_Corrected_(3) (3).ipynb
│   └── presentation_yolo_broad_category_training.ipynb
├── assets/              # Static assets (logos, images)
├── tests/               # Unit tests
│   ├── test.py
│   └── test_model.py
└── requirements.txt     # Python dependencies
```

## Development

Run the app in development mode with auto-reload:

```bash
streamlit run main.py --server.runOnSave true
```

## Model Training

Check the `notebooks/` directory for:
- Dataset preprocessing pipeline
- YOLOv8 training notebook with broad category classification
