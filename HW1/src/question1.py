from pprint import pprint
import numpy as np

from mainlib.file import load_img_2D, save_img_from_array
from mainlib.threshold import threshold
from mainlib.morph_operations import dilation, erosion

from other.loading import execute

######## QUESTION 1 ########

def question1_test():

    src_img = load_img_2D('test4.png')
    threshold(src_img, 100)

    struct_el = np.ones((3, 3)).astype(np.uint8) * 255

    dilated_img = dilation(src_img, struct_el)
    eroded_img = erosion(src_img, struct_el)

    # Output
    save_img_from_array(dilated_img, out_name = 'dilated_out_img.png')
    save_img_from_array(eroded_img, out_name = 'eroded_out_img.png')


# Executing test
execute(question1_test)



def arr_test():
    temp = np.array([(0, 0, 0, 0, 0, 0, 0, 0),
                     (1, 1, 1, 1, 1, 1, 1, 0),
                     (0, 0, 0, 1, 1, 1, 1, 0),
                     (0, 0, 0, 1, 1, 1, 1, 0),
                     (0, 0, 1, 1, 1, 1, 1, 0),
                     (0, 0, 0, 1, 1, 1, 1, 0),
                     (0, 0, 1, 1, 0, 0, 0, 0),
                     (0, 0, 0, 0, 0, 0, 0, 0)
                    ]) * 255

    str_el = np.ones((3, 3)) * 255

    pprint(temp)
    print("\n")
    pprint(dilation(temp, str_el))
    print("\n")
    pprint(erosion(temp, str_el))
