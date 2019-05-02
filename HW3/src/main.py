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

print(images[0])

print()
#
# image = images[0]
#
# cv2.imwrite('img.png', image)
#
# from skimage.segmentation import mark_boundaries
#
# # My implementation
# labels = slic(image, 100)
# seg = mark_boundaries(image, labels)
# cv2.imwrite('labels.png',  seg)
#
# print(seg)
#
#
# # From module
# from skimage.segmentation import slic
#
# labels2 = slic(image, 100)
# seg2 = mark_boundaries(image, labels2)
# cv2.imwrite('labels2.png', seg2)
#
#
# print(seg2)

# from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

# load the image and convert it to a floating point data type
image = images[1]


# loop over the number of segments
numSegments = 600
# apply SLIC and extract (approximately) the supplied number
# of segments
segments = slic(image, numSegments, 20)

# show the output of SLIC
fig = plt.figure("Superpixels -- %d segments" % (numSegments))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(image, segments))
plt.axis("off")

cv2.imwrite('labels.png', segments)

# show the plots
plt.show()