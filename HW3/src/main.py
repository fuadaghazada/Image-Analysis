import cv2

from src.slic_segmentation import slic
from src.load_data import load_images

'''
	Main script for testing

	@author: Fuad Aghazada
	@date: 2/5/19
	@version: 1.0.0
'''

# Loading all images in the dataset
images, _ = load_images()


image = images[0]

cv2.imwrite('img.png', image)


# My implementation
labels = slic(image, 500, 10, num_iterations=20)
cv2.imwrite('labels.png',  labels)

# From module
from skimage.segmentation import slic

labels2 = slic(image, 500)
cv2.imwrite('labels2.png', labels2)