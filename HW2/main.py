import os
import sys
import time

import cv2
import numpy as np

from src.load import load_images
from src.stich import stich_images

from src.detect_describe import detect_describe_local_features, describe_raw_pixel_based, subimage
from pprint import pprint

# Menu
print("--------------------------------")
print("----Image Stitching Software----")
print("--------------------------------")

print("Notes: \n--Please put your txt file into '/txt' directory")
print("--Please put your images into '/data' directory and have a .txt file containing name of the images in it\n\n")

print("Please enter the name of txt file (with file extension): ", end='')

txt_name = str(input())
images, status = load_images(txt_name)

if status is False:
    print("* No such a file inside the directory! \n* Make sure your file name is correct and in proper directory.\n\n")
    sys.exit()

# --------------------------------------------------------------------------------------------------

print("\n\n")
print("Please select a descriptor: (Enter number)")
print("0. SIFT descriptor")
print("1. Raw-pixel based descriptor")

choice = str(input())

if choice != "0" and choice != "1":
    print("Invalid choice! Please choose one of the choices with number")
    sys.exit()

print("\nPlease wait until the stitching process is finished. It may take a while...\n\n")
res = stich_images(images, int(choice))

# --------------------------------------------------------------------------------------------------

print("\n\nProcess is finished! Do you want to save the result image? (y/n): ", end='')

choice = str(input()).lower()

if choice != "y" and choice != "n":
    print("Invalid choice! Please choose y or n")
    sys.exit()
elif choice == "y":
    cv2.imwrite('saved/result' + str(int(time.time())) + ".png", res)

print("\n\nShowing the image (Press any key to close...)\n\n")
cv2.imshow('Result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n\nBye!")
