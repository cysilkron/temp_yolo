from utils.tools.type_checker import is_segmap
from utils.tools.image import bbox_from_mask, crop_bbox_area_mask, get_segmap, draw_mask
from boxx import show as boxx_show
class MaskedImage:
    def __init__(self, img, mask, cls_id=None) -> None:
        self.img = img
        self.mask = mask
        self.cls_id= cls_id
        
        self.im_size = self.img.shape
        self.mask_size = self.mask.shape
        self._bbox = None
        self._segmap = None
        self._cropped_img = None
        self._cropped_mask = None

    @property
    def segmap(self):
        if not is_segmap(self._segmap):
            self._segmap = get_segmap(self.img, self.mask)
        return self._segmap

    @property
    def bbox(self):
        if self._bbox is None:
            self._bbox = bbox_from_mask(self.mask.astype('bool'))
        return self._bbox

    @property
    def cropped_img(self):
        if self._cropped_img is None:
            self._cropped_img, self._cropped_mask,  = crop_bbox_area_mask(self.mask.astype('bool'), self.img)
        return self._cropped_img

    @property
    def cropped_mask(self):
        if self._cropped_mask is None:
            self._cropped_mask,  = crop_bbox_area_mask(self.mask.astype('bool'))
        return self._cropped_mask


    def show_both(self, **kwargs):
        boxx_show(self.img, self.mask, **kwargs)

    def draw_mask(self):
        return draw_mask(self.img, self.mask)

    def show_drawn(self, **kwargs):
        boxx_show(self.draw_mask(), **kwargs)