from pprint import pprint
import numpy as np

from operations import dilation, erosion, threshold
from file import load_img_2D, save_img_from_array

######## QUESTION 2 ########
src_img = load_img_2D('sonnet.png')


strc_el = np.array([(255, 255, 255),
                    (255, 255, 255),
                    (255, 255, 255)])



threshold(src_img, 180)
# out_img = erosion(src_img, strc_el)
# out_img = dilation(out_img, strc_el)


save_img_from_array(src_img, out_name = 'sonnet_out.png')
