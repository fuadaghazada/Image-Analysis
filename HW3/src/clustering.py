import cv2
import numpy as np

'''
	Applying K-Means algorithm in the dataset

	@author: Fuad Aghazada
	@date: 5/5/19
	@version: 1.0.0
'''

'''
	K-Means algorithm
'''


def apply_k_means(samples, num_of_clusters, images, images_labels, super_pixel_count):

	# Parameters (other than input parameters)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)		# Criteria for termination of algorithm
	attempts = 10																	# Number of times the algorithm is executed using different labels
	flags = cv2.KMEANS_RANDOM_CENTERS												# How initial centers are taken

	# Applying the K-Means
	_, k_labels, k_centers = cv2.kmeans(samples, num_of_clusters, None, criteria, attempts, flags)

	# Clusters for all images
	clusters = []

	# Getting the cluster matrices for each image
	for i, image in enumerate(images):
		img_label = images_labels[i]
		h, w = img_label.shape

		cluster = np.zeros((h, w), dtype=np.uint8)
		counter = 0

		for ii in range(h):
			for jj in range(w):
				label = img_label[ii, jj]
				cluster_id = k_labels[label + counter]
				cluster[ii, jj] = cluster_id

		clusters.append(cluster)
		counter += super_pixel_count[i]

	return clusters