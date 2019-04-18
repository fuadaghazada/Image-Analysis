import cv2
from match import get_matching_points, match, draw_matches
from detect_describe import detect_describe_local_features

'''
    Registering/Aligning two images

    @param: image1 - the first image
    @param: image2 - the second image
    @param: keypoints1 - the interest points of image1
    @param: keypoints2 - the interest points of image2
    @param: matches - matching indexes for keypoints
    @return: img - resgitered/aligned image
'''
def register_images(image1, image2, keypoints1, keypoints2, matches):

    # Matching points of two images
    m_kps0, m_kps1 = get_matching_points(keypoints1, keypoints2, matches)

    # Finding Homography using RANSAC
    homography, status = cv2.findHomography(m_kps0, m_kps1, cv2.RANSAC, 4.0)

    # Creating result image
    img = cv2.warpPerspective(image1, homography, (image1.shape[1] + image2.shape[1], image1.shape[0]))
    img[0:image2.shape[0], 0:image2.shape[1]] = image2

    return img

'''
    Stiching two images and returning them as an image

    @param: image1 - the first image
    @param: image2 - the second image
    @threshold: threshold - for matching distance
    @return: result, image - result image after stiching and image with matching lines
'''
def stich_two_images(image1, image2, threshold = 100):

    # Features (Keypoints and Descriptors)
    feature1 = detect_describe_local_features(image1)
    feature2 = detect_describe_local_features(image2)

    # Keypoints
    keypoints1 = feature1['keypoints']
    keypoints2 = feature2['keypoints']

    # Descriptors
    descriptor1 = feature1['descriptors']
    descriptor2 = feature2['descriptors']

    # Matching indices between two images
    matches = match(descriptor1, descriptor2, threshold)

    # Image with matching lines and result image
    image = draw_matches(image1, image2, keypoints1, keypoints2, matches)
    result = register_images(image1, image2, keypoints1, keypoints2, matches)

    return result, image

'''
    Stiching given number of images:

    Note: Make sure the images has been ordered in a correct way

    @param: images - list of images
    @param: dir: 1 if the images go left / 0 if right
    @return: result - stiched images
'''
def stich_images(images, dir = 0):

    # Change direction of the image list
    if dir == 1:
        images = images[::-1]

    # Result
    result = images[0]

    for i in range(1, len(images)):
        result, image = stich_two_images(result, images[i])

    return result
