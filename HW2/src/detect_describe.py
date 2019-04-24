import math
import cv2
import numpy as np

# Initializing SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

'''
    Detecting Local features (Interest points)
    using SIFT detector for the image

    @param: images - the given image
    @return: interest_points - 2D list of interest of points for each image
'''
def detect_local_features(image):

    print("Detecting keypoints...")

    interest_points = sift.detect(image, None)

    print("Keypoints detected!\n--")

    return interest_points

'''
    Describing the local features using SIFT descriptor & Raw pixel based
    the given the images

    @param: images - the given list of images
    @return: features - tuple of keypoints and descriptors for each image
'''
def detect_describe_local_features(image):

    print("Computing SIFT keypoints & descriptors...")

    (keypoints, descriptor) = sift.detectAndCompute(image.astype('uint8'), None)
    features = {"keypoints": keypoints, "descriptors": descriptor}

    print("SIFT Keypoints & Descriptors computed!\n--")

    return features

'''
    Raw Pixel based descriptor

    @param: image = the given image
    @return: features - tuple of keypoints and descriptors for each image
'''
def describe_raw_pixel_based(image):

    # Detecting keypoints
    keypoints = detect_local_features(image)

    descriptor = []
    for keypoint in keypoints:
        # Properties of keypoint
        coordinates = keypoint.pt
        size = keypoint.size
        orientation = keypoint.angle

        # Square around keypoint with given coordinate, orientation and scale
        img = np.copy(image)
        square = subimage(img, coordinates, orientation, int(size) , int(size))

        # Calculating the histogram
        histogram = cv2.calcHist([square], [0], None, [256], [0, 256])
        histogram = [x[0] for x in histogram]
        descriptor.append(histogram)

    return {"keypoints": keypoints, "descriptors": np.asarray(descriptor, dtype = 'float32')}

'''
    Reference: https://stackoverflow.com/questions/11627362/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
'''
def subimage(image, center, theta, width, height):

    theta *= math.pi / 180 # convert to rad

    v_x = (math.cos(theta), math.sin(theta))
    v_y = (-math.sin(theta), math.cos(theta))
    s_x = center[0] - v_x[0] * ((width-1) / 2) - v_y[0] * ((height-1) / 2)
    s_y = center[1] - v_x[1] * ((width-1) / 2) - v_y[1] * ((height-1) / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image, mapping, (width, height), flags = cv2.WARP_INVERSE_MAP, borderMode = cv2.BORDER_REPLICATE)
