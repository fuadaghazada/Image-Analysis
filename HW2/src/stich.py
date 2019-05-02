import sys
import cv2
import numpy as np

from src.match import get_matching_points, match, draw_matches
from src.detect_describe import detect_describe_local_features, describe_raw_pixel_based

'''
    Registering/Aligning two images

    @param: image1 - the first image
    @param: image2 - the second image
    @param: keypoints1 - the interest points of image1
    @param: keypoints2 - the interest points of image2
    @param: matches - matching indexes for keypoints
    @return: img - resgitered/aligned image
'''
def register_images(image1, image2, keypoints1, keypoints2, matches, blend_type = 0):

    # Matching points of two images
    m_kps0, m_kps1 = get_matching_points(keypoints1, keypoints2, matches)

    # Finding Homography using RANSAC
    homography, status = cv2.findHomography(m_kps0, m_kps1, cv2.RANSAC)

    # Creating result image
    img = cv2.warpPerspective(image1, homography, (image1.shape[1] + image2.shape[1], image1.shape[0]))

    # ---------------------------------------
    img = blend(img, image1, image2, blend_type)

    return img

'''
    Blending the overlapping area of the images

    @param: img - the affined version for image1
    @param: image1 - the first image
    @param: image2 - the second image
    @return: result image
'''
def blend(img, image1, image2, blend_type):
    # BLENDING
    # Finding Overlap
    x1, y1, x2, y2 = find_overlap(img, image2)

    d = x2 - x1
    im1 = image1[:, 0:d]
    im2 = image2[:, x1:x2]

    # ---------------------------------------
    # Adding second image to the result
    img[0:image2.shape[0], 0:image2.shape[1]] = image2

    # Averaging with equal weights and unequal weights
    if blend_type == 0:
        avg = im1 * 0.5 + im2 * 0.5
    else:
        i = 0
        inc = 1
        avg = np.zeros((im1.shape))
        while i < d:
            avg[:, i] = im1[:, i] * (1 - inc) + im2[:, i] * inc
            inc -= (1 / d)
            i += 1

    img[y1:y2, x1:x2] = avg

    return img

'''
    Finding overlapped region (not perfect) of two images

    @param: img1 -  the first image
    @param: img2 -  the second image
'''
def find_overlap(img, img2):
    L = len(img)
    x1_up = 0
    for i in range(0, L):
        if img[20, i] != 0:
            x1_up = i
            break
        img[0, i] = 255

    x1_down = 0
    for i in range(0, L):
        if img[img.shape[0] - 20, i] != 0:
            x1_down = i
            break
        img[img.shape[0] - 1, i] = 255

    x1 = min(x1_up, x1_down)
    x2 = img2.shape[1]
    y1 = 0
    y2 = img.shape[0]

    return x1, y1, x2, y2

'''
    Returning features of two images and their match points

    @param: image1 - the first image
    @param: image2 - the second image
    @param: threshold - for matching distance
    @param: desc_type - type of descriptor: 0 for SIFT , 1 for Raw based
    @return:
'''
def get_features_of_two_images(image1, image2, desc_type = 0):

    if desc_type == 0:
        # Features (Keypoints and Descriptors)
        feature1 = detect_describe_local_features(image1)
        feature2 = detect_describe_local_features(image2)
    else:
        # Raw-based descriptor
        feature1 = describe_raw_pixel_based(image1)
        feature2 = describe_raw_pixel_based(image2)

    # Keypoints
    keypoints1 = feature1['keypoints']
    keypoints2 = feature2['keypoints']

    # Descriptors
    descriptor1 = feature1['descriptors']
    descriptor2 = feature2['descriptors']

    # Matching indices between two images
    matches = match(descriptor1, descriptor2)

    return {"image1": (keypoints1, descriptor1),
            "image2": (keypoints2, descriptor2),
            "matches": matches }

'''
    Stiching two images and returning them as an image

    @param: image1 - the first image
    @param: image2 - the second image
    @param: desc_type = description type
    @param: blend_type = blend type
    @return: result, image - result image after stiching and image with matching lines
'''
def stich_two_images(image1, image2, desc_type, blend_type):

    # Features (Keypoints and Descriptors) & matches
    features = get_features_of_two_images(image1, image2, desc_type)

    # Keypoints
    keypoints1 = features['image1'][0]
    keypoints2 = features['image2'][0]

    # Descriptors
    descriptor1 = features['image1'][1]
    descriptor2 = features['image2'][1]

    # Matching indices between two images
    matches = features['matches']

    # Image with matching lines and result image
    image = draw_matches(image1, image2, keypoints1, keypoints2, matches)
    result = register_images(image1, image2, keypoints1, keypoints2, matches, blend_type)

    return result, image

'''
    Stiching given number of images:

    Note: Make sure the images has been ordered in a correct way

    @param: images - list of images
    @param: dir: 1 if the images go left / 0 if right
    @return: result - stiched images
'''
def stich_images(images, desc_type = 0, blend_type = 0):

    # Determining the direction of panorama
    dir = get_direction(images)
    if dir == -1:
        print("Improper images!")
        sys.exit()

    # Change direction of the image list
    if dir == 1:
        images = images[::-1]

    # Result
    result = images[0]

    for i in range(1, len(images)):
        result, image = stich_two_images(result, images[i], desc_type, blend_type)

    return result

'''
    Determine the direction of the panorama

    @return: dir: 1 if the images go left / 0 if right / -1 all images the same (rare)
'''
def get_direction(images):

    for i in range(len(images) - 1):
        dir1, dir2 = direction(images[0], images[1]), direction(images[1], images[0])
        if dir1 > dir2:
            return 1
        elif dir1 < dir2:
            return 0
    return -1

'''
    For determining the direction of the image in the panorama
    (Helper function)

    @param: image1 - the first image
    @param: image2 - the second image
'''
def direction(image1, image2):

    # Features (Keypoints and Descriptors) & matches
    features = get_features_of_two_images(image1, image2)

    # Keypoints
    keypoints1 = features['image1'][0]
    keypoints2 = features['image2'][0]

    # Descriptors
    descriptor1 = features['image1'][1]
    descriptor2 = features['image2'][1]

    # Matching indices between two images
    matches = features['matches']

    # Matching points of two images
    m_kps0, m_kps1 = get_matching_points(keypoints1, keypoints2, matches)

    # Finding Homography using RANSAC
    homography, status = cv2.findHomography(m_kps0, m_kps1, cv2.RANSAC)

    # Creating result image
    img = cv2.warpPerspective(image1, homography, (image1.shape[1] + image2.shape[1], image1.shape[0]))

    # Calculating the percentage of black area in the affined image
    all_0_count = list(img.flatten()).count(0)
    black_region_0_count = list(img[0:image2.shape[0], image2.shape[1]:].flatten()).count(0)
    percentage = (black_region_0_count / all_0_count) * 100

    return percentage
