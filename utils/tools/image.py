from .general import *
from .type_checker import is_bool_mask, is_2d_img, is_color_img, is_grey_img

from imageio import imread
from imgaug import imshow
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from imgaug import parameters as iap

# img in this file is always expected as np.ndarray

def img_info(img, title=None):
    if title:
        dash(len(title))
    if not isinstance(img, np.ndarray):
        raise TypeError('must be np.ndarray')
    
    print(f"img.shape:  {img.shape}")
    print(f"img.dtype:  {img.dtype}")
    print(f"np.unique(img):  {np.unique(img)}")


def read_color_img(img_path):
    img = imread(img_path)
    if img.ndim != 3:
        raise ValueError('Only color img allowed')

    return img


def bgr2rgb_(img):
    return img[:, :, ::-1].transpose(2, 0, 1)


def rgb2bgr_(img):
    return img[:, :, ::-1].transpose(2, 0, 1)


# --bboxes--
def get_bbs(img: np.ndarray, labels: np.ndarray, has_cls_id=False):
    """
    labels = [[cls_id, x1, y1, x2, y2]]
    """
    if has_cls_id:
        bbs = [BoundingBox(*bbox, label=cls_id) for cls_id, *bbox in labels]
    else:
        bbs = [BoundingBox(*bbox) for bbox in labels]
    return BoundingBoxesOnImage(bbs, shape=img.shape)


def draw_bb(img, labels, size=3, has_cls_id=False):
    bbs = get_bbs(img, labels, has_cls_id=has_cls_id)
    return bbs.draw_on_image(img, size=size)


def show_bb(img, bbox, **kwargs):
    imshow(draw_bb(img, bbox, **kwargs))


def bbox_from_mask(bool_mask):
    mask_y, mask_x = np.array(np.where(bool_mask))
    x1 = np.min(mask_x)
    y1 = np.min(mask_y)
    x2 = np.max(mask_x) + 1
    y2 = np.max(mask_y) + 1

    return [x1, y1, x2, y2]

def img_height_width(img):
    is_2d_img(img, warn=True)

    if is_color_img(img):
        h, w, _ = img.shape
    elif is_grey_img(img):
        h, w = img.shape

    return h, w

# --segmentatinon mask--
def get_segmap(img, bool_mask):
    return SegmentationMapsOnImage(bool_mask, shape=img.shape)

#todo: save as png (jpg may face the changing value in np.uint8)
def save_mask(mask, dest, fname):
    fname = dest / (Path(fname).stem + ".npy")
    with open(str(fname), "wb") as f:
        np.save(f, mask)


def read_mask(fpath):
    with open(str(fpath), "rb") as f:
        mask = np.load(f)
    return mask


def draw_mask(img, mask):
    segmap = get_segmap(img, mask)
    return segmap.draw_on_image(img)[0]


def show_mask(img, mask):
    imshow(draw_mask(img, mask))


def get_mask_indices(mask):
    i, j = np.where(mask)
    indices = tuple(np.meshgrid(np.arange(min(i), max(i) + 1),
                        np.arange(min(j), max(j) + 1),
                        indexing='ij'))
    return indices

def crop_bbox_area_mask(mask, img=None):

    mask_indices = get_mask_indices(mask)
    if img is not None:
        ih, iw = img_height_width(img)
        mh, mw = img_height_width(mask)
        if ih == mh and iw == mw:
            return img[mask_indices], mask[mask_indices]
        else:
            raise ValueError(f'Width or Height or Image: {img.shape}, do not match with mask: {mask.shape}')

    return mask[mask_indices]

def mask_area(bool_mask):
    is_bool_mask(bool_mask, warn=True)
    return bool_mask.sum()

# -- distribution --
def stochastic_params(a_min=0.3, a_max=0.7, dist="uniform"):
    options = ["uniform", "normal"]

    if dist == "uniform":
        return iap.Clip(
            iap.Uniform(iap.Normal(a_min + 0.05, 0.01), iap.Normal(a_max - 0.05, 0.01)),
            a_min,
            a_max,
        )
    elif dist == "normal":
        mean = (a_min + a_max) / 2
        return iap.Normal(mean, 0.05)

    else:
        raise ValueError(f"only dist in {options} is allowed")


def show_dist(params):
    iap.show_distributions_grid(params)