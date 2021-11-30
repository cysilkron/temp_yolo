import numpy as np
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def is_array(x, warn=False):

    if not isinstance(x, np.ndarray):
        if warn:
            raise TypeError("variable has to be numpy ndarray")
        else:
            return False
    return True


# --- image ---
def is_color_img(img, warn=False):
    if not is_array(img, warn):
        return False

    if not (img.ndim == 3 and img.shape[-1] == 3):
        if warn:
            raise ValueError("Color Image has to be in shape of (h, w, 3)")
        else:
            return False
    return True


def is_grey_img(arr, warn=False):
    is_array(arr)
    if not arr.ndim == 2:
        if warn:
            raise ValueError("Array has to be in shape of (h, w)")
        else:
            return False
    return True


def is_2d_img(img: np.ndarray, warn=False):
    if is_color_img(img) or is_grey_img(img):
        return True
    else:
        if warn:
            raise ValueError("invalid 2 dimension numpy Image")
        return False


# --- mask ----
is_mask = is_grey_img


def is_bool_mask(mask, warn=False):
    if is_mask(mask) and mask.dtype == bool:
        return True

    else:
        if warn:
            raise TypeError("dtype in mask must be boolean")
        
    return False


def is_segmap(segmap, warn=False):
    if not isinstance(segmap, SegmentationMapsOnImage):
        if warn:
            raise TypeError("`segmap` is not an instance of `SegmentationMapsOnImage`")
        else:
            return False
    return True


# --- bbox ----
def is_bbox(bbox, warn=False):
    if len(bbox) != 4:
        if warn:
            raise ValueError("len of bbox should be 4")
        else:
            return False
    return True


def is_bboxes(bboxes, warn=False):
    bboxes = np.array(bboxes)
    if not (bboxes.ndim == 2 and bboxes.shape[-1] == 4):
        if warn:
            raise ValueError("bboxes should be an array of (n, 4)")
        else:
            return False
    return True


def is_bbs(bbs, warn=False):
    if not isinstance(bbs, BoundingBoxesOnImage):
        if warn:
            raise TypeError("`bbs` is not an instance of `BoundingBoxesOnImage`")
        else:
            return False
    return True


# -------- decorator ----------
# '''example implementations to decorate the type checkers'''
#
# // raw implementation
#    ------------------
# def is__color_img(arg_idx: int, key:str = None):
#     def accept_func(func):
#         def accept_args(*args, **kwargs):
#             img = args[arg_idx]
#             is_color_img(img)
#             func(*args, **kwargs)
#         return accept_args
#     return accept_func
#
# // implement with decorator module
#    -------------------------------
# from decorator import decorator
#
# @decorator
# def is__color_img(func, idx=None, key=None, *args, **kw):
#     is_color_img(args[idx])
#     func(*args, **kw)
#
#  // implement with dec_checker
#     --------------------------
# is__color_img_ = dec_checker(is_color_img)
#


def dec_checker(checker_func):
    """example of using decorated type checker
    is__color_img = dec_checker(is_color_img)

    @is__color_img(0)
    @is__bboxes(1)
    def get_bbs(img, bboxes)
        return BoundingBoxesOnImage(bboxes, shape=img.shape)
    """

    def accept_idx(arg_idx: int):
        def accept_func(func):
            def accept_args(*args, **kwargs):
                to_check = args[arg_idx]
                checker_func(to_check, warn=True)
                return func(*args, **kwargs)

            return accept_args

        return accept_func

    return accept_idx


is__color_img = dec_checker(is_color_img)
is__grey_img = dec_checker(is_grey_img)
is__2d_img = dec_checker(is_2d_img)
is__mask = dec_checker(is_mask)
is__segmap = dec_checker(is_segmap)
is__bbox = dec_checker(is_bbox)
is__bboxes = dec_checker(is_bboxes)
is__bbs = dec_checker(is_bbs)


# '''example of using decorated type checker'''
# @is__color_img(0)
# @is__bboxes(1)
# def get_bbs(img, bboxes)
#     return BoundingBoxesOnImage(bboxes, shape=img.shape)