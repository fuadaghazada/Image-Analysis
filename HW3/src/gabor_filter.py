import cv2
import numpy as np

'''
	Applying Gabor filter on image 

	@author: Fuad Aghazada
	@date: 4/5/19
	@version: 1.0.0
'''

'''
	Extracting Gabor features from a given image its super pixel labels
	
	:param kernel_size - dimension (size) of the filter
	:param sigma - standard deviation of Gaussian function
	:param theta - orientation 
	:param lambda - wavelength of sinusoidal factor
	:param gamma - spatial aspect ratio
	:param psi - phase offset
	:param ktype - type of Gabor filter
	
	:return result_images - images after applying gabor filter with different parameters
	:return average_gabors - average of Gabor features
'''


def gabor_feature_extractions(image, labels):
	# Converting the input image to gray scale
	gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Orientation parameters
	orientations = [30, 50, 70, 90]

	# Wavelength parameters
	wavelengths = [2, 4, 6, 8]

	# Kernel parameters
	kernel_size = 31
	sigma = 4.0
	gamma = 0.5
	psi = 0

	# Number of labels
	num_labels = np.max(labels) + 1

	# Results images after applying Gabor averages
	result_images = []

	# Gabor averages
	average_gabors = np.zeros((num_labels, 16))

	for i, wavelength in enumerate(wavelengths):
		for j, orientation in enumerate(orientations):

			# Applying the Gabor filter with different parameters
			kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, orientation, wavelength, gamma, psi)
			res = cv2.filter2D(gray_scale, cv2.CV_8UC3, kernel)
			result_images.append({'image': res, 'wavelength': wavelength, 'orientation': orientation})

			# Average Gabor for each super pixel
			gabor_average = gabor_average_superpixel(res, labels)
			for ii in range(num_labels):
				average_gabors[ii, 4 * (i - 1) + j] = gabor_average[ii]

	return result_images, average_gabors


'''
	Gabor average for each super pixel
	
	:param gabor_image - result image after applying Gabor filter on it
	:param labels - labels of super pixels of image
'''


def gabor_average_superpixel(gabor_image, labels):

	# Dimensions of labels
	l_height, l_width = labels.shape

	# Number of labels
	num_labels = np.max(labels) + 1

	# Count of labels and Sum for Gabor values
	label_count = [0] * num_labels
	gabor_sum = [0] * num_labels

	# Iterating over labels
	for i in range(l_height):
		for j in range(l_width):
			label = labels[i, j]
			label_count[label] += 1

			gabor_sum[label] += gabor_image[i, j]

	# Calculating the average
	avg_gabor = [0] * num_labels

	for i in range(num_labels):
		if label_count[i] != 0:
			avg_gabor[i] = gabor_sum[i] / label_count[i]

	return avg_gabor

