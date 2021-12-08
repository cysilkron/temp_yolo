from pathlib import Path
from typing import List

from utils.tools.general import dash
from .load_datasets import LoadProduct, LoadNoise
import imgaug.random as iarandom
import imgaug.augmenters as iaa
from utils.tools.image import get_segmap
from utils.tools.datasets import MaskedImage
from functools import partial
from utils.paster.augment import adjust_hsv

class LoadProductWithNoise:
    def __init__(self, rng_seed=1234) -> None:
        # front_img_dpath = Path('/mnt/disks/lightbox_hand_eccsd-copy')
        '''jump'''
        front_img_dpath = Path('/mnt/disks/datasets')



        self.load_lightbox(front_img_dpath/'lightbox_china_train'/'clean_coco.json', front_img_dpath)
        self.load_hand(front_img_dpath/'cropped_hand_14k'/'clean_coco.json', front_img_dpath)
        self.load_eccsd(front_img_dpath/'eccsd'/'clean_coco.json', front_img_dpath)

        if isinstance(rng_seed, int):
            self.rng = iarandom.RNG(rng_seed)
        elif isinstance(rng_seed, iarandom.RNG):
            self.rng = rng_seed
        else:
            raise TypeError('Invalid type for rng_seed')

        self.choosen_dataset = []

    def load_lightbox(self, json_path, img_root_dir=None):
        corrupted_imgs = ['lightbox_china_train/images/h45__wangwang_coco_orange/0000028.jpg',
                    'lightbox_china_train/images/h45__wangwang_coco_orange/0000029.jpg',
                    'lightbox_china_train/images/h45__wangwang_coco_orange/0000030.jpg']

        self.lb_coco = LoadProduct(json_path, img_root_dir)
        self.lb_coco.filter_front_img(corrupted_imgs)
        self.lb_coco.set_weighted_img_df()
        self.lb_coco.set_angle_weights({
                    't45': 1,
                    'tb45': 1,
                    't90': 0,
                    'tb90': 0,
                    'h45': 1,
                    'hb45': 1,
                    'not_found': 1
                })
        # self.lb_coco.plot_category_sample(3, 3000, title='before set cat weight')
        self.lb_coco.set_cat_weights(sigma=2)
        self.lb_coco.set_sample_weight()
        # self.lb_coco.plot_category_sample(3, 3000, title='after set cat weight')

    
    def load_eccsd(self, json_path, img_root_dir=None):
        self.eccsd_coco = LoadNoise(json_path, img_root_dir)
        self.eccsd_coco.set_sample_weight()

    def load_hand(self, json_path, img_root_dir=None):
        self.hand_coco = LoadNoise(json_path, img_root_dir)
        self.hand_coco.set_sample_weight()

    
    def transform_with_mask(self, aug_sequence: List[iaa.Sequential], img, mask, rng_seed=None):
        segmap = get_segmap(img, mask)
        aug_img, aug_segmap = iaa.Sequential(aug_sequence, seed=rng_seed)(image=img, segmentation_maps=segmap)
        return aug_img, aug_segmap.get_arr()


    def next_batch_and_transform(self, coco, aug_seq, bs=1) -> MaskedImage:
        fnames, imgs, masks, cls_ids = coco.next_batch(bs)
        # mask_imgs = [MaskedImage(img, mask, cls_id) for img, mask, cls_id in zip(imgs, masks, cls_ids)]
        mask_imgs = []
        for img, mask, cls_id in zip(imgs, masks, cls_ids):
            aug_img, aug_mask = self.transform_with_mask(aug_seq, img, mask, rng_seed=self.rng)
            mask_imgs.append(MaskedImage(aug_img, aug_mask, cls_id))

        return mask_imgs

    def next_batch_with_multiple_dataset(self, bs=None, min_rand_item=1, max_rand_item=5):
        '''
        bs: int
            batch_size for item to generate, if batch_size is set, `min_rand_item` and `max_rand_item` will be ignored


        '''


        lb_coco_aug = [
            iaa.Rotate((0,360)),
            iaa.Resize((0.5, 0.7)),
            adjust_hsv()
        ]
        

        hand_coco_aug = [
            # iaa.Rotate((-90,90)),
            # iaa.Rotate((-45,45)),
            iaa.Resize(0.5),
        ]

        eccsd_coco_aug = [
            iaa.Rotate((-45,45)),
            iaa.Resize((640,480)),
            adjust_hsv()
        ]
        
        lb_transform = partial(self.next_batch_and_transform, self.lb_coco, lb_coco_aug)
        hand_transform = partial(self.next_batch_and_transform, self.hand_coco, hand_coco_aug)
        eccsd_transform = partial(self.next_batch_and_transform, self.eccsd_coco, eccsd_coco_aug)

        batch_functions = [
            lb_transform,
            hand_transform,
            eccsd_transform
        ]

        item_per_img = bs if bs else self.rng.randint(min_rand_item, max_rand_item)
        choosen_func_idxs = self.rng.choice(3, item_per_img, p=[0.65, 0.175, 0.175])
        
        # print(f"choosen_func:  {choosen_func}")
        # aug_imgs, aug_masks = list(zip([fn() for fn in batch_functions]))

        self.choosen_dataset += list(choosen_func_idxs)

        all_masked_imgs = []
        try:
            for idx in choosen_func_idxs:
                batch_masked_imgs = batch_functions[idx](bs=1)
                if len(batch_masked_imgs):
                    all_masked_imgs.append(batch_masked_imgs[0])
        except IndexError as e:
            dash()
            print(f"choosen_func_idxs:  {choosen_func_idxs}")
            dash()
            raise e
    
        return all_masked_imgs