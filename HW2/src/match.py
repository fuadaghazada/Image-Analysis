import cv2
import math
import numpy as np

'''
	Calculating the Euclidean distance between two keypoints

	@param: desc1 - the first descriptor
	@param: desc2 - the second descriptor
	@return: distance - the Euclidean distance
'''
def calc_eculidean_distance(desc1, desc2):
	# Calculating the distance
	distance = np.sqrt(np.sum((desc1[:, np.newaxis, :] - desc2[np.newaxis, :, :]) ** 2, axis = -1))

	return distance

'''
	Finding matching keypoints between two images given their descriptors

	@param: desc1 - descriptor of the first image
	@param: desc2 - descriptor of the second image
	@return: matches - matching indices for matching points
'''
def match(desc1, desc2, threshold = 100):

	print("Computing matching keypoints...")

	# For keeping the mathcing points
	matches = []

	# Number of keypoints for each image
	num_keypoints1 = desc1.shape[0]
	num_keypoints2 = desc2.shape[0]

	# Calculating the distances between keypoints of two images
	distances = calc_eculidean_distance(desc1, desc2)

	for i in range(num_keypoints1):
		for j in range(num_keypoints2):
			if distances[i][j] < threshold:
				matches.append((i, j))

	print("Matching points computed!\n--")

	return matches

'''
	Drawing the lines between the matching points of two images
	--DEBUG proposely--

	@param: image1 - the first input image
	@param: image2 - the second input image
	@param: keypoints1 - interest points for the first image
	@param: keypoints2 - interest points for the second image
	@param: matches - matching points between the two images
'''
def draw_matches(image1, image2, keypoints1, keypoints2, matches):
	# Dimensions of the images
	h1, w1 = image1.shape
	h2, w2 = image2.shape

	# 2 images horizontally together
	img = np.zeros((max(h1, h2), w1 + w2), dtype = "uint8")
	img[0:h1, 0:w1] = image1
	img[0:h2, w1: ] = image2

	# Drawing the lines
	for i in range(len(matches)):
		# Matching points
		pt1 = (int(keypoints1[matches[i][0]].pt[0]), int(keypoints1[matches[i][0]].pt[1]))
		pt2 = (int(keypoints2[matches[i][1]].pt[0]) + w1, int(keypoints2[matches[i][1]].pt[1]))

		# Drawing mathcing points
		cv2.circle(img, pt1, 2, (0, 255, 0), 1)
		cv2.circle(img, pt2, 2, (0, 0, 255), 1)

		# Drawing line between the points
		cv2.line(img, pt1, pt2, (255, 0, 0), 1)

	return img

'''
	Given the keypoints of the each image and matching indices returnin the matching keypoints

	@param: keypoints1 - interest points for the first image
	@param: keypoints2 - interest points for the second image
	@param: matches: matching indices for the keypoints
	@return: m_keypoints1, m_keypoints2 - matching keypoints
'''
def get_matching_points(keypoints1, keypoints2, matches):
	# List for keeping the results
	m_keypoints1 = []
	m_keypoints2 = []

	for match in matches:
		m_keypoints1.append(keypoints1[match[0]].pt)
		m_keypoints2.append(keypoints2[match[1]].pt)

	m_keypoints1 = np.asarray(m_keypoints1, dtype = np.float32)
	m_keypoints2 = np.asarray(m_keypoints2, dtype = np.float32)

	return m_keypoints1, m_keypoints2
