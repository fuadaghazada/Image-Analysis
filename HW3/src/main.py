import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb

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

# ------- Parameters --------
num_of_super_pixels = 200
num_of_clusters = 3
compactness = 10
wavelengths = [2, 4, 6, 8]
orientations = [30, 50, 70, 90]
# ---------------------------

# For keeping labels of each image in dataset
images_labels = []

print("-- Parameters --")
print("\tNumber of Super pixels:", num_of_super_pixels)
print("\tCompactness:", compactness)
print("\tWavelengths (for Gabor filtering):", *wavelengths)
print("\tOrientations (for Gabor filtering):", *orientations)
print("\tNumber of clusters:", num_of_clusters)
print("\n********************* \n")

# -- Part 1 --

print("--- PART 1 ---")

for i, image in enumerate(images):
	# Segmenting
	labels = slic(image, num_of_super_pixels, compactness)
	segmented_image = mark_boundaries(image, labels)

	# Keeping labels
	images_labels.append(labels)

	print(f"\tSLIC algorithm is applied for image {i}")

	# Saving results
	plt.imsave('../output/slic/segmented' + str(i) + '.png', segmented_image)

print("\n********************* \n")

# For keeping all average Gabors
all_data = None
super_pixel_count = []

print("--- PART 2 ---")

# -- Part 2 --

for i, image in enumerate(images):
	# Gabor features and average gabors
	result_images, average_gabors = gabor_feature_extractions(image, labels, wavelengths, orientations)

	print(f"\tGabor filter extraction for image {i}")

	# Saving Gabor results
	for img in result_images:
		cv2.imwrite('../output/gabor/image' + str(i) + '_wavelen' + str(img['wavelength']) + "_orient" + str(img['orientation']) + '.png', img['image'])

	# Concatenating average Gabors
	if all_data is None:
		all_data = average_gabors
	else:
		all_data = np.concatenate((all_data, average_gabors), axis=0)

	# Keeping super pixel counts
	super_pixel_count.append(average_gabors.shape[0])

print("\n********************* \n")

print("--- PART 3 ---")

# -- Part 3 --

# Applying the K-Means
clusters = apply_k_means(np.float32(all_data), num_of_clusters, images, images_labels, super_pixel_count)

print(f"\tApplying K-means...")

# Saving the clusters
for i, cluster in enumerate(clusters):
	false_color_img = label2rgb(cluster, images[i], image_alpha=0)
	plt.imsave('../output/cluster/cluster' + str(i) + '.png', false_color_img)

print("\n***** Processes finished! *****\n")


