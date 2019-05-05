import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

from src.load_data import load_images
from src.gabor_filter import gabor_feature_extractions
from src.clustering import apply_k_means

'''
	Main script for steps in assignment

	@author: Fuad Aghazada
	@date: 2/5/19
	@version: 1.0.0
'''

# Loading all images in the dataset
images, _ = load_images()
image = images[1]

# Parameters
num_of_super_pixels = 1000
compactness = 15

# For keeping labels of each image in dataset
images_labels = []

# -- Part 1 --
for i, image in enumerate(images):
	# Segmenting
	labels = slic(image, num_of_super_pixels, compactness)
	segmented_image = mark_boundaries(image, labels)

	# Keeping labels
	images_labels.append(labels)

	# Saving results
	plt.imsave('../output/slic/segmented' + str(i) + '.png', segmented_image)

# For keeping all average Gabors
all_feature_matrices = []

# -- Part 2 --
for i, image in enumerate(images):
	# Gabor features and average gabors
	result_images, average_gabors = gabor_feature_extractions(image, labels)

	# Saving Gabor results
	for img in result_images:
		cv2.imwrite('../output/gabor/image' + str(i) + '_wavelen' + str(img['wavelength']) + "_orient" + str(img['orientation']) + '.png', img['image'])

	all_feature_matrices.append(average_gabors.flatten())

labels, centers = apply_k_means(np.asarray(all_feature_matrices), 9)

centers = np.uint8(centers)
res = centers[labels.flatten()]
res2 = res.reshape(image.shape)

cv2.imwrite('../output/clustering.png', res2)


