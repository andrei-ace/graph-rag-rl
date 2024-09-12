import re
import pytesseract


def ocr_elements(image_pil, items):
    elements = []
    for item in items:
        x0, y0, x1, y1 = map(int, item["bbox"])
        label = item["label"]
        label_id = item["label_id"]
        cropped_img = image_pil.crop((x0, y0, x1, y1))
        text = pytesseract.image_to_string(cropped_img)
        text = re.sub(r'[^\x20-\x7E]', '', text)  # Keep only printable ASCII characters        
        elements.append(((x0, y0, x1, y1), text, label, label_id))
    return elements