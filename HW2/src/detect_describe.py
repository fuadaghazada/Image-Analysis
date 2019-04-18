import cv2

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
    @param: keypoints - the list of keypoints for the image
'''
def describe_raw_pixel_based(image, keypoints):

    for keypoint in keypoints:
        # Properties of keypoint
        coordinates = keypoint.pt
        scale = keypoint.size
        orientation = keypoint.angle
