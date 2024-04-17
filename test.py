from LucasKanade import LucasKanadeInverse
from PIL import Image
import numpy as np

def scale_image_to_255(image):
    # Normalize the image to [0, 1]
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val) if max_val > min_val else image - min_val

    # Scale to [0, 255]
    scaled_image = (normalized_image * 255).astype(np.uint8)
    return scaled_image

I = Image.open("woman.png")
if I.mode != "L":
    I = I.convert("L")
R = Image.open("woman_distort.png")
if R.mode != "L":
    R = R.convert("L")

eps = 0.01
i_max = 1000
LK = LucasKanadeInverse(I, R, eps, i_max)
LK.run()
# print(images.shape)
# image1 = scale_image_to_255(images[:, :, 2])
# print(image1.shape)
# image1 = Image.fromarray(image1, mode = "L")
# image1.show()


