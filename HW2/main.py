import os
import cv2
import numpy as np


from src.load import load_images
from src.stich import stich_images

from src.detect_describe import detect_describe_local_features, describe_raw_pixel_based, subimage
from pprint import pprint

images = load_images("goldengate.txt")
res = stich_images(images, 0)

cv2.imshow('Result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
