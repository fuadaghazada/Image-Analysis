from pprint import pprint
import numpy as np

from operations import dilation, erosion, threshold
from file import load_img_2D, save_img_from_array

######## TEST ########
src_img = load_img_2D('test0.png')
threshold(src_img, 100)

struct_el = np.array([(255, 255, 255, 255, 255),
                      (255, 255, 255, 255, 255),
                      (255, 255, 255, 255, 255),
                      (255, 255, 255, 255, 255),
                      (255, 255, 255, 255, 255)])

dilated_img = dilation(src_img, struct_el)
eroded_img = erosion(src_img, struct_el)


# # Output
save_img_from_array(dilated_img, out_name = 'dilated_out_img.png')
save_img_from_array(eroded_img, out_name = 'eroded_out_img.png')


# struct_el = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)])
# test_img_arr = np.array([(0, 0, 0, 0, 0, 0, 0, 0),
#                          (1, 1, 1, 1, 1, 1, 1, 0),
#                          (0, 0, 0, 1, 1, 1, 1, 0),
#                          (0, 0, 0, 1, 1, 1, 1, 0),
#                          (0, 0, 1, 1, 1, 1, 1, 0),
#                          (0, 0, 0, 1, 1, 1, 1, 0),
#                          (0, 0, 1, 1, 0, 0, 0, 0),
#                          (0, 0, 0, 0, 0, 0, 0, 0)
# ])
# out_img = erosion(test_img_arr, struct_el)
# pprint(out_img)
