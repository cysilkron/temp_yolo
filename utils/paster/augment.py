import imgaug.augmenters as iaa
from utils.tools.general import *


def color_chnl_adjust(chnl, values):

    return iaa.WithChannels(
        chnl,
        iaa.Add(values),
    )

def adjust_hsv():
    hyp_dict = {
        'hsv_h': [0.8, [-4, 4]],
        'hsv_s': [0.8, [-15, 20]],
        'hsv_v': [0.8, [-20, 30]]
    }


    return iaa.WithColorspace(
        to_colorspace="HSV",
        from_colorspace="RGB",
        children=iaa.Sequential(
            [
                iaa.Sometimes(
                    hyp_dict["hsv_h"][0],  # chance
                    color_chnl_adjust(0, hyp_dict["hsv_h"][1]),  # (min, max)
                ),
                iaa.Sometimes(
                    hyp_dict["hsv_s"][0],
                    color_chnl_adjust(1, hyp_dict["hsv_s"][1]),
                ),
                iaa.Sometimes(
                    hyp_dict["hsv_v"][0],
                    color_chnl_adjust(2, hyp_dict["hsv_v"][1]),
                ),
            ]
        ),
    )



#---motion blur---
def thicken(mask_edge, iterations=2):
    kernel = np.ones((3,3), np.uint8)
    return cv2.dilate(mask_edge.astype('uint8'), kernel, iterations=iterations)

def int_bb_arr(bb):
    '''get int array of BoundingBox'''
    return [bb.x1_int, bb.y1_int, bb.x2_int, bb.y2_int]

def resolve_shape_rounding(a1, a2):
    def until_shape(big_a, small_a):
        return big_a[:small_a.shape[0], :small_a.shape[1]]

    def solve_rounding(a1, a2, idx):
        if a1.shape[idx] > a2.shape[idx]:
            a1 = until_shape(a1, a2)
        elif a1.shape[idx] < a2.shape[idx]:
            a2 = until_shape(a2, a1)
        return a1, a2
    
    if a1.ndim >= 2 and a2.ndim>=2:
        #resolve arr size different
        a1, a2 = solve_rounding(a1, a2, 0)
        a1, a2 = solve_rounding(a1, a2, 1)                    
    return a1, a2
