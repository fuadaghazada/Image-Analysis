import os
import sys
import cv2

from src.config import ROOT

'''
    Loading images from the given directory to a list.

    @param: filename - name of the file contains the list of the image names
    @return: images - list of images in the directory
'''
def load_images(filename, scale = 2):
    dir = ROOT + '/data/'
    images = []
    files = []

    with open(ROOT + '/' + filename, 'r') as file:
        for line in file:
            files.append(line.replace('\n', ''))

    # Sorting the file accroding ot their names
    files = sorted(files)

    # Loading the images into a list
    for file in files:
        img = cv2.imread(dir + file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
        images.append(img)

    return images
