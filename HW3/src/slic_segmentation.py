import math
import cv2
import numpy as np

import warnings
warnings.filterwarnings('ignore')

'''
	Simple Linear Iterative Clustering (SLIC) algorithm implementation

	@author: Fuad Aghazada
	@date: 2/5/19
	@version: 1.0.0
'''

'''
	SLIC algorithm

	:param image - the input image for being segmented
	:param k - amount of super pixels 
	:param m - constant for stabilizing colors in the distance (compactness)
'''


def slic(image, k, m=10):
	# Properties
	h, w, _ = image.shape			# Image dimension
	N = w * h						# Number pixels in image
	S = int(math.sqrt(N / k))		# Sampling interval

	# Converting the image from RGB to LAB
	image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

	# Initializing cluster centers in lowest gradient positions (3x3)
	cluster_centers = init_super_pixels(image_lab, S)

	# Initializing labels and distances
	labels = np.ones((h, w), dtype=np.uint8) * (-1)
	distances = np.ones((h, w)) * math.inf

	# -- ASSIGNMENT -- (Part 1)

	# TODO: Repeat until E <= threshold
	for i in range(10):
		# Iterating over the cluster centers
		for c, center in enumerate(cluster_centers):
			distances, labels = calc_distances(center, image_lab, distances, labels, S, m, w, h, c)

	return labels


'''
	Setting the labels according to the distance calculations
	
	:param center - center of the cluster
	:param image_lab - input image in LAB color space
	:param distances - distances between the pixels of image and center of cluster
	:param labels - labels for each pixel of the input image (show to which cluster the pixel belongs)
	:param S - sampling interval
	:param m - constant for stabilizing colors in the distance (compactness)
	:param w - width of the input image
	:param h - height of the input image
	:param c - index of the center of cluster in the list of cluster - cluster id

	Reference: https://github.com/jayrambhia/superpixels-SLIC/blob/master/SLICcv.py
'''


def calc_distances(center, image_lab, distances, labels, S, m, w, h, c):
	l, a, b, x, y = center

	x1 = x - S if x - S > 0 else 0
	x2 = x + S if x + S <= w else w

	y1 = y - S if y - S > 0 else 0
	y2 = y + S if y + S <= h else 0

	# --- CALCULATIONS using loops (slow) ---

	'''
	# Calculating the distance between the image and cluster center
	for i in range(x1, x2):
		for j in range(y1, y2):
			# Color distance
			d_lab = np.sqrt(np.sum(np.square(image_lab[j, i] - np.asarray([l, a, b]))))

			# Pixel distance
			d_pix = math.sqrt((x - i) ** 2 + (y - j) ** 2)

			# Distance between the (i, j) pixel and center
			distance = math.sqrt(d_lab ** 2 + ((d_pix * m) / S) ** 2)

			if distance < distances[j, i]:
				distances[j, l] = distance
				labels[j, i] = c
	'''

	# --- ------------------------------------ ---

	# --- CALCULATIONS using matrix operations (fast) ---

	# Region of 2S x 2S
	region_2s = image_lab[y1: y2, x1: x2]

	# Color distance
	d_lab = np.sqrt(np.sum(np.square(image_lab[y, x] - region_2s)))

	# Pixel distance
	yy, xx = np.ogrid[y1: y2, x1: x2]

	d_pix = ((yy - y) ** 2 + (xx - x) ** 2) ** 0.5

	# Distance between the (i, j) pixel and center
	distance = (d_lab ** 2 + ((d_pix * m) / S) ** 2) ** 0.5

	# Applying the same condition: distance < distance[j, i]
	distance_2s = distances[y1: y2, x1: x2]
	cond = distance < distance_2s
	distance_2s[cond] = distance[cond]
	distances[y1: y2, x1: x2] = distance_2s

	# Setting the labels according to the condition above
	labels[y1: y2, x1: x2][cond] = c

	# --- ------------------------------------ ---

	return distances, labels


'''
	Initializing the super pixels
	
	:param image - the input image for being segmented (in LAB color space)
	:param S - sampling interval
'''


def init_super_pixels(image, S):
	h, w, _ = image.shape

	# Initializing cluster centers
	cluster_centers = []

	for j in range(S, h - (S // 2), S):
		for i in range(S, w - (S // 2), S):
			x, y = calc_min_gradient_pos((i, j), image)
			lab = image[y, x]
			cluster_centers.append([*lab, x, y])

	return np.asarray(cluster_centers)


'''
	Calculating minimum gradient position the given super pixel center
	in 3x3 neighborhood
	
	:param sp_center - center of the given super pixel
	:param img_lab - image in CIELAB color space
'''


def calc_min_gradient_pos(sp_center, img_lab):
	# Initializing the minimum gradient to Infinity
	min_gradient = math.inf
	min_position = sp_center

	sp_x, sp_y = sp_center

	for j in range(sp_y - 1, sp_y + 2):
		for i in range(sp_x - 1, sp_x + 2):
			center_i_j = img_lab[j, i][0]
			center_i_j1 = img_lab[j, i + 1][0]
			center_i1_j = img_lab[j + 1, i][0]

			# Calculating the gradient (wrt L)
			gradient = math.sqrt((center_i_j - center_i1_j) ** 2 + (center_i_j - center_i_j1) ** 2)

			if gradient < min_gradient:
				min_gradient = abs(center_i_j - center_i1_j) + abs(center_i_j - center_i_j1)
				min_position = (i, j)

	# print("Min position:", min_position)
	return min_position