import numpy as np
from PIL import Image, ImageFilter, ImageOps

def preprocess(image_path: str) -> np.ndarray:
    
    img = Image.open(image_path)
    img = img.convert("L")
    img = img.resize((300, 300))
    img = ImageOps.equalize(img)
    img = img.point(lambda x: 255 if x > 128 else 0)
    img = img.filter(ImageFilter.FIND_EDGES)

    return np.array(img)
