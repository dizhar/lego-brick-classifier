from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
CLASS_NAMES_PATH = PROJECT_ROOT / "data" / "class_names.json"
EXAMPLES_DIR = PROJECT_ROOT / "data" / "examples"

# Model parameters
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# App settings
PAGE_TITLE = "LEGO Classifier"
PAGE_ICON = "🧱"
TOP_K_PREDICTIONS = 5

# Example images configuration
EXAMPLE_IMAGES = {
    'simple': {
        'path': EXAMPLES_DIR / 'example_simple.png',
        'label': 'Simple Scene',
        'description': 'Few pieces, easy detection'
    },
    'complex': {
        'path': EXAMPLES_DIR / 'example_complex.png',
        'label': 'Complex Scene',
        'description': 'Many pieces, crowded'
    },
    'mixed': {
        'path': EXAMPLES_DIR / 'example_mixed.png',
        'label': 'Mixed Types',
        'description': 'Various piece types'
    }
}