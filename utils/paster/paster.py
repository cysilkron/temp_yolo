from collections import defaultdict
from typing import Tuple

from IPython.core.display import display
from utils.tools.datasets import MaskedImage
from utils.tools.type_checker import is_color_img, is_bool_mask
from utils.tools.image import img_height_width, mask_area
import numpy as np

# for debugger
import utils.tools.image as iu
from utils.tools.general import all_len_is_same, arr_map, dash
import boxx
import cv2
import pandas as pd
from utils.paster.create_datasets import LoadProductWithNoise
class PasteMaskedImage:
    def __init__(self, back_img, save_mask=False):
        if not is_color_img(back_img):
            raise ValueError('expect RGB color img')
        self.back_img = back_img
        self._save_mask = save_mask
        self.paste_sequence = 0
        self.back_mask = None
        if self._save_mask:
            self.back_mask = np.zeros_like(back_img[:,:,0])
             # 0 is background in mask, max sequence is 255 as uint8 array
            self.visible_mask_areas = []

    def _verify_save_mask(self):
        if not self._save_mask:
            raise ValueError('save_mask is False, cant perform related operations')

    def _paste_visible_mask_on_back_mask(self, visible_mask, bbox):

        self._verify_save_mask()
        x1, y1, x2, y2 = bbox
        
        self.paste_sequence += 1
        if self.back_mask[y1:y2, x1:x2].shape == visible_mask.shape:
            self.back_mask[y1:y2, x1:x2][visible_mask.astype('bool')] = self.paste_sequence
        else:
            print('shape not equal in paste_visible_mask_on_back_mask')

    def find_visible_mask_on_back_img(self):
        '''used to find the visible_mask after sequence of pasting, only support 255 instance to be pasted for uint8 image '''
        self._verify_save_mask()
        return [(self.back_mask == idx) for idx in range(1, self.paste_sequence+1)] #start at 1, as 0 is mask background


    def cal_visible_mask_areas(self):
        self.visible_mask_areas = [mask_area(m) for m in self.find_visible_mask_on_back_img()]

    
    def paste_object(self, front_img, front_mask, coord: Tuple, bbox_from_visible=False):

        if not (len(front_mask) and is_bool_mask(front_mask)):
            raise ValueError('front mask must be boolean mask')

        x_coord, y_coord = coord

        src_h, src_w, _ = front_img.shape
        dst_h, dst_w, _ = self.back_img.shape

        x_offset, y_offset = x_coord-int(src_w/2), y_coord-int(src_h/2)
        y1, y2 = max(y_offset, 0), min(y_offset + src_h, dst_h)
        x1, x2 = max(x_offset, 0), min(x_offset + src_w, dst_w)
        y1_m = 0 if y1>0 else -y_offset
        x1_m = 0 if x1>0 else -x_offset
        y2_m = src_h if y2<dst_h-1 else dst_h - y_offset 
        x2_m = src_w if x2<dst_w-1 else dst_w - x_offset

        bbox = []
            
        if y1_m>=src_h or x1_m>=src_w or y2_m<0 or x2_m<0:
            return self.back_img, bbox, None, None


        visible_back = self.back_img[y1:y2, x1:x2]
        visible_front = front_img[y1_m:y2_m, x1_m:x2_m]
        
        visible_mask = front_mask[y1_m:y2_m, x1_m:x2_m]
        visible_back[visible_mask] = visible_front[visible_mask]
        bbox = [x1,y1,x2,y2]


        if self._save_mask:
            self._paste_visible_mask_on_back_mask(visible_mask, bbox)
        

        return self.back_img, bbox, visible_front, visible_mask




class DebugPaster(PasteMaskedImage):
    def __init__(self, back_img, debug=True, save_mask=False):
        super().__init__(back_img, save_mask=save_mask)
    
        self.visible_masks = []
        self.visible_fronts = []
        self.front_mask_areas = []
        
        self.cls_ids = []
        self.bboxes = []
        self.visible_bboxes = []
        self.labels = []
        self.labels_before_clean = []

        self.is_noise_classes = []
        self.is_empty_bboxes = []
        self.is_visible_too_lows = []
        self.bbox_to_remove = []

        self.debug = debug
        self.hyp_dict__overlap_thereshold = 0.35
        self.init_debug_info()

    def create_labels(self):
        if not all_len_is_same(self.cls_ids, self.bboxes):
            raise ValueError('len of cls_ids and bboxes should be equal')

        labels = np.zeros((len(self.visible_bboxes), 5))

        self.labels = np.zeros((len(self.bboxes), 5)) #[cls_id, *xyxy]
        if len(self.bboxes):
            self.labels[:, 0] = self.cls_ids
            self.labels[:, 1:] = self.bboxes

    def _verify_labels(self):
        if not all_len_is_same(self.cls_ids, self.bboxes, self.is_visible_too_lows):
            raise ValueError('len of all arguments should be equal')

        for cls_id, bbox in zip(self.cls_ids, self.bboxes):
            self.is_noise_classes.append(is_noise_class(cls_id))
            self.is_empty_bboxes.append(not len(bbox))

        flags = [np.array(flag) for flag in [self.is_noise_classes, self.is_empty_bboxes, self.is_visible_too_lows]]

        #label to remove
        self.bbox_to_remove = flags[0] | flags[1] | flags[2]

        self.debug_info['is_noise_class'] = self.is_noise_classes
        self.debug_info['empty bbox'] = self.is_empty_bboxes
        self.debug_info['is_visible_too_lows'] = self.is_visible_too_lows
        self.debug_info['bbox_to_remove'] = self.bbox_to_remove

    def clean_labels(self):
        self.check_is_visible_to_low()
        self._verify_labels()
        self.labels_before_clean = self.labels.copy()
        self.labels = self.labels[~self.bbox_to_remove]

        self.debug_info['labels before clean'] = self.labels_before_clean
        self.debug_info['labels after clean'] = self.labels
        self.debug_info['bbox_to_remove'] = self.bbox_to_remove

    def show_area(self):
        print(f"front_mask area before paste: {self.front_mask_areas}")
        print(f"original pasted visible mask_area: {[mask_area(m) for m in self.visible_masks]}")
        print(f"real visible_mask_area after overlapped by paste: {self.visible_mask_areas}")

    def check_is_visible_to_low(self):
        self._verify_save_mask()
        self.cal_visible_mask_areas()
        self.init_area_debug_info()

        for i, mask_area_after_overlapped in enumerate(self.visible_mask_areas):
            mask_area_before_paste = self.front_mask_areas[i]
            area_pct_after_paste = remaining_area_pct(mask_area_before_paste, mask_area_after_overlapped) 
            self.is_visible_too_lows.append(area_pct_after_paste < self.hyp_dict__overlap_thereshold)

            self.save_area_debug_info(i, mask_area_after_overlapped)
        
    def find_visible_bboxes(self):
        self._verify_save_mask()
        for i, m in enumerate(self.find_visible_mask_on_back_img()):
            self.visible_bboxes.append(
                iu.bbox_from_mask(m) if m.any() else self.bboxes[i])
        self.debug_info['visible_bboxes'] += self.visible_bboxes            
    
    def print_diff_mask_area(self):
        print(f"front_mask area before paste: {self.front_mask_areas}")
        print(f"original pasted visible mask_area: {[mask_area(m) for m in self.visible_masks]}")
        print(f"real visible_mask_area after overlapped by paste: {self.visible_mask_areas}")

    def init_area_debug_info(self):
        self.area_debug_info = defaultdict(list)

    def init_debug_info(self):
        self.debug_info = defaultdict(list)

    def get_debug_info(self):
        # return pd.DataFrame(self.debug_info)
        class_cols = ['cls_id', 'is_noise_class']
        lbl_cols = ['labels before clean', 'labels after clean', 'bboxes', 'visible_bboxes']
        mask_area_cols = ['front_mask_areas', 'mask_area_after_overlapped', 'pct']
        bbox_remove_cols = ['empty bbox', 'is_visible_too_lows', 'is_noise_class', 'bbox_to_remove']
        all_cols = [*class_cols, *lbl_cols, *mask_area_cols, *bbox_remove_cols]


        '''jump'''
        self.debug_info = {**self.debug_info, **self.area_debug_info}

        #orient index to fill NaN for unequal len of list
        return pd.DataFrame.from_dict(self.debug_info, orient='index').T[all_cols].T

    def show_debug_info(self):
        display(self.get_debug_info())


    def save_area_debug_info(self, i, mask_area_after_overlapped):
        self.area_debug_info['cls_id'].append(self.cls_ids[i])
        self.area_debug_info['pct'].append(remaining_area_pct(self.front_mask_areas[i], mask_area_after_overlapped))
        self.area_debug_info['front_mask_areas'].append(self.front_mask_areas[i])
        self.area_debug_info['mask_area_after_overlapped'].append(mask_area_after_overlapped)
        self.area_debug_info['are_visible_too_lows'].append(self.is_visible_too_lows[i])

    def display_area_debug_info(self):
        display(pd.DataFrame(self.area_debug_info))

    def print_area_debug_info(self, i, mask_area_after_overlapped):
        print(f"cls_id:  {self.labels_before_clean[i][0]}")
        print(f'pct {i}: {remaining_area_pct(self.front_mask_areas[i], mask_area_after_overlapped)}')
        print(f'area {i}: {[self.front_mask_areas[i], mask_area_after_overlapped]}')
        print(f"are_visible_too_lows[i]:  {self.are_visible_too_lows[i]}")
        dash()

    def batch_pasting_flow(self, n, coco: LoadProductWithNoise, show_pasted_mask=False, show_debug_info=False, show_area_debug_info=False):

        for i, masked_img in enumerate(coco.next_batch_with_multiple_dataset(n)):

            cropped_img, cropped_mask = masked_img.cropped_img, masked_img.cropped_mask

            bool_mask = (cropped_mask > 0)
            if not (is_color_img(cropped_img) and is_bool_mask(bool_mask)):
                raise ValueError('invalid img or mask')

            if not bool_mask.any():
                raise ValueError('Emtpy Instance in Mask')
            
            self.front_mask_areas.append(mask_area(bool_mask))

            point = RandomCoordOnImage(self.back_img).point
            back_img, bbox, visible_front, visible_mask = self.paste_object(cropped_img, bool_mask, point)

            self.bboxes.append(bbox)
            self.cls_ids.append(masked_img.cls_id)
            self.visible_masks.append(visible_mask.astype('bool'))
            self.visible_fronts.append(visible_front)

        

        self.debug_info['bboxes'] = self.bboxes
        self.create_labels()
        self.clean_labels()

        self.find_visible_bboxes()



        labels = np.zeros((len(self.visible_bboxes), 5))

        if len(self.visible_bboxes):
            labels[:, 0] = np.array(self.cls_ids)
            labels[:, 1:] = np.array(self.visible_bboxes)
            labels = labels[~self.bbox_to_remove]

        self.labels = labels

        if show_area_debug_info:
            self.print_diff_mask_area()

        if show_pasted_mask and self.debug:
            self.draw_pasted_mask()

    def draw_pasted_mask(self):
        drawn_labels_on_mask = draw_sequence_mask(self.back_mask, self.labels)
        if not np.array_equal(self.visible_bboxes, self.bboxes):
            visible_bboxes = np.array(self.visible_bboxes)[~np.array(self.bbox_to_remove)]
            drawn_visible_bbox_on_mask = draw_sequence_mask(self.back_mask, visible_bboxes, has_cls_id=False)
            boxx.show(drawn_labels_on_mask, self.back_img, drawn_visible_bbox_on_mask)
        else:
            boxx.show(drawn_labels_on_mask, self.back_img)


class PasteOnLabeledImage():
    def __init__(self, coco, debug=True, save_mask=True):
        self.debug = debug
        self.save_mask = save_mask
        self.coco = coco

    def _paste_on_labeled_img(self, back_img, labels, **kwargs):
            '''paste the segmented front_img into back_img with existing labeled bounding boxes
            labels should be in [class_id, x1, y1, x2, y2] format

            back_img: np.ndarray
                background img to be pasted with front_img. 
                only color img (H x W x 3)is allowed, this issue is supposed to be fixed in future.

            labels: np.ndarray | torch.tensor
                labels should be in [class_id, x1, y1, x2, y2] format with the shape of N x 5
            '''


            db_paster = DebugPaster(back_img, save_mask=True, debug=self.debug)
            db_paster.batch_pasting_flow(n=None, coco=self.coco, **kwargs)
            labels = None # ignore previous labels input

            # back_img, labels = self.paste_front_imgs(back_img, num2gen=1)

            # return db_paster.back_img, labels
            return db_paster.back_img, db_paster.labels


class RandomCoordOnImage:
    def __init__(self, image:np.ndarray):
        h,w = img_height_width(image)
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        self.point = (x,y)

def remaining_area_pct(mask_area_before_paste: int, mask_area_after_paste: int):
    return 1 - (mask_area_before_paste - mask_area_after_paste) / mask_area_before_paste

def is_noise_class(cls_id):
    return int(cls_id) < 0


def stack_grey_img_to_3chnl(grey_img):
    return np.stack([grey_img for i in range(3)]).transpose(1,2,0)

def draw_sequence_mask(back_mask, labels, copy=True, has_cls_id=True):
    if copy:
        back_mask = back_mask.copy()
    hsv_mask = cv2.cvtColor(stack_grey_img_to_3chnl(back_mask.astype('uint8')), cv2.COLOR_RGB2HSV)
    value_count = len(np.unique(hsv_mask))
    brightness_ratio = 255 // value_count
    hsv_mask[:,:, 1] = hsv_mask[:,:, 1] * brightness_ratio
    hsv_mask[:,:, 2] = hsv_mask[:,:, 2] * brightness_ratio
    rgb_mask = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2RGB)
    # print(f"img_info(draw_sequence_mask):  {iu.img_info(rgb_mask)}")
    return iu.draw_bb(rgb_mask, labels, has_cls_id=has_cls_id)
