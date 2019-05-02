import os
import cv2

'''
	Loading the data images

	@author: Fuad Aghazada
	@date: 2/5/19
	@version: 1.0.0
'''

DIR = '../data/'

'''
	Data loader
'''


def load_images(directory=DIR):

	images = []
	status = False

	try:
		for file in os.listdir(directory):
			image = cv2.imread(directory + file, 1)

			# Validation (non image files)
			if image is not None:
				images.append(image)

		status = True
	except Exception as e:
		print("---ERROR---\n" + str(e) + "\n-----------")
		status = False

	return images, status
