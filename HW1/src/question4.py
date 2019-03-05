import numpy as np

from mylib.file import load_img_2D, save_img_from_array
from mylib.ffiltering import gaussian_filter

######## TEST ########
src_img = load_img_2D('test1.png')

# Applying Gaussian filter with the given sigma
out_img = gaussian_filter(src_img, 9)

# Output
save_img_from_array(out_img.astype(np.uint8), out_name = 'glp_out.png')
