import cv2

'''
	Applying K-Means algorithm in the dataset

	@author: Fuad Aghazada
	@date: 5/5/19
	@version: 1.0.0
'''

'''
	K-Means algorithm
'''


def apply_k_means(samples, num_of_clusters):

	# Parameters (other than input parameters)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)		# Criteria for termination of algorithm
	attempts = 10																	# Number of times the algorithm is executed using different labels
	flags = cv2.KMEANS_RANDOM_CENTERS												# How initial centers are taken

	# Applying the K-Means
	compactness, labels, centers = cv2.kmeans(samples, num_of_clusters, None, criteria, attempts, flags)

	return labels, centers