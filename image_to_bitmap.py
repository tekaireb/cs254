from PIL import Image
import numpy as np


def image_to_bitmap(img_path):
    img = Image.open(img_path)
    arr = np.array(img)

    # Split into RGB channels
    r, g, b = np.split(arr, 3, axis=2)
    r = r.reshape(-1)
    g = r.reshape(-1)
    b = r.reshape(-1)

    # Convert to grayscale
    bitmap = list(map(lambda x: 0.299*x[0]+0.587*x[1]+0.114*x[2],
                      zip(r, g, b)))
    bitmap = np.array(bitmap).reshape([arr.shape[0], arr.shape[1]])

    return bitmap


bitmap = image_to_bitmap('./images/desert_road_100x67.jpg')
im = Image.fromarray(bitmap.astype(np.uint8))
im.save('images/desert_road_input.bmp')
