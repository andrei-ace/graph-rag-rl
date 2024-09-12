import os
import warnings
import requests
from ultralytics import YOLO


CLASS_NAMES = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title"
]

def download_model_weights(url, save_path):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Weights already exist at {save_path}")


weights_url = (
    "https://github.com/moured/YOLOv10-Document-Layout-Analysis/releases/download/doclaynet_weights/yolov10x_best.pt"
)
weights_path = "models/yolov10x_best.pt"

download_model_weights(weights_url, weights_path)
warnings.filterwarnings(
    "ignore", message="You are using `torch.load` with `weights_only=False`", module="ultralytics.nn.tasks"
)
# Load the YOLOv10 model
model = YOLO(weights_path)


def detect_layout_elements(image_pil, model=model):
    results = model(image_pil, conf=0.2, iou=0.8)
    items = []
    for r in results:
        classes = r.names
        for box in r.boxes:
            items.append({
                "label": classes[box.cls.item()],
                "label_id": int(box.cls.item()),
                "bbox": box.xyxy.tolist()[0],
            })
    return items