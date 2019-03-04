import numpy as np

from file import load_img_2D, save_img_from_array
from operations import threshold
from filtering import convole

######## TEST ########
src_img = load_img_2D('test0.png')
threshold(src_img, 100)

filter = np.array([(-1, -2, -1),
                   (0, 0, 0),
                   (1, 2, 1)
])
conv_img = convole(src_img, filter)

# Output
save_img_from_array(conv_img, out_name = 'conv_out_img.png')
