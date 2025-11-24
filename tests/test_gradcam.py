import numpy as np
from ultralytics import YOLO

from src.gradcam import YOLOGradCAM


def test_gradcam_runs():
    model = YOLO("models/best.pt")
    cam = YOLOGradCAM(model)
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    out = cam.generate(img)
    assert isinstance(out, np.ndarray)
    assert out.shape == (480, 640, 3)
    assert out.dtype == np.uint8