import os
import cv2
import numpy as np

import const
from load import load_images
from detect_describe import detect_describe_local_features
from stich import stich_images, stich_two_images

from pprint import pprint


images = load_images(const.DIR_GOLDENGATE)
res = stich_images(images)

cv2.imshow('Result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
