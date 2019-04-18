import os
import cv2

'''
    Loading images from the given directory to a list.

    @param: dir - the given directory for the images
    @return: images - list of images in the directory
'''
def load_images(dir, scale = 2):
    images = []
    files = []
    for file in os.listdir(dir):
        files.append(file)

    files = sorted(files)
    for file in files:
        img = cv2.imread(dir + file, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
        images.append(img)

    return images
