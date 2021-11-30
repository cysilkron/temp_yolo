import numpy as np
import pandas as pd
from utils.tools.general import Path, List
from utils.tools.image import read_color_img
from utils.tools.datasets import MaskedImage

from pycocotools.coco import COCO
import boxx

from utils.tools.general import dash

class Coco2Df:
    def __init__(self, coco_path):
        self.set_coco(coco_path)
        self.set_coco_df()
    
    def set_coco(self, coco_path):
        self.coco_path = Path(coco_path)
        assert (
            self.coco_path.is_file() and self.coco_path.suffix == ".json"
        ), f"`{coco_path}` is not a valid json file"

        self.coco = COCO(coco_path)
        

    def set_coco_df(self):
        # coco_json = boxx.loadjson(self.coco_path)
        self.coco_df = {}
        self.img_df = self.coco_df["images"] = pd.DataFrame(self.coco.imgs).T
        self.anns_df = self.coco_df["annotations"] = pd.DataFrame(self.coco.anns).T
        self.cat_df = self.coco_df["categories"] = pd.DataFrame(self.coco.cats).T

class LoadCocoImage(Coco2Df):
    def __init__(self, coco_path, img_root_dir=None):
        '''
        Parameters:
        -----------
        coco_path: str | Path
            path to coco format json file
        img_root_dir: str | Path (optional)
            path to root directory containing all the image files. If not provided, 
            it will set to the same directory with the coco json file
        '''
        super().__init__(coco_path)
        self.img_root_dir = img_root_dir if img_root_dir else self.coco_path.parent
        if self.img_root_dir.is_dir():
            self.set_front_fnames(self.img_root_dir)
        else:
            raise ValueError('img_root_dir not a valid directory')

        self.filter_front_img([]) # start filter for first time to remove file not found in self.front_fnames
        # self.set_hyp_dict(hyp_dict)
        # self.reset_augs()
        # self.reset_data_dist()

    def set_front_fnames(self, img_root_dir):
        self.front_fnames = [
            fname for fname in self.coco_df['images']['file_name']
            if (img_root_dir / fname).is_file()
        ]

    def filter_front_img(self, corrupted_imgs):
        """filter img on coco_df
        Args:
        ----
        corrupted_imgs: List[str], optional
            filtered the corrupted img that match the `file_name` field in coco_df
        Eg:
        [ 'h45__wangwang_coco_orange/0000028.jpg',
        'h45__wangwang_coco_orange/0000029.jpg']
        """

        if len(corrupted_imgs):
            is_corrupted = self.coco_df["images"]["file_name"].isin(corrupted_imgs)
            self.coco_df["images"] = self.coco_df["images"][~is_corrupted]

        fname_is_exist = self.coco_df["images"]["file_name"].isin(self.front_fnames)
        id_is_match = self.coco_df["images"]["id"].isin(self.anns_df["image_id"])
        img_is_chosen = fname_is_exist & id_is_match
        self.coco_df["images"] = self.coco_df["images"][img_is_chosen]
        self.coco_df["images"].reset_index(inplace=True, drop=True)
        self.img_df = self.coco_df["images"]

    def get_mask_by_idx(self, idx):
        return self.anns_df.iloc[idx]["mask"]

    def get_mask_by_image_id(self, img_id):
        ann = self.coco.loadAnns(ids=[int(img_id)])[0]
        return self.coco.annToMask(ann)

    def get_img_fname(self, img_id):
        is_id = self.coco_df["images"]["id"] == img_id

        if is_id.any():
            return self.coco_df["images"][is_id]["file_name"].values[0]
        else:
            return None

    def get_anns_by_img_ids(self, img_ids: List[int]):
        '''img_id: only list of int accepted, cant be array'''
        img_ids = np.array(img_ids).astype('int')
        ann_ids = self.coco.getAnnIds(imgIds=list(img_ids))
        return self.coco.loadAnns(ann_ids)

    def get_catId_by_image_id(self, img_id):
        is_id = self.anns_df["image_id"] == img_id
        return self.anns_df[is_id]["category_id"].values[0]

    def get_batch_img_ids(self, n, by="ann"):
        options = ["ann", "image"]

        if by == "ann":
            return self.anns_df["image_id"].sample(n)
        elif by == "image":
            return self.coco_df["images"]["id"].sample(n)
        else:
            raise ValueError(f"only {options} allowed")

    def next_batch(self, n, create_obj=False):
        '''the annotation receord should be find by `get_batch_img_ids` as all the filter logic
         will be done in the image_df
        Return:
        -------
        - file_names : List[Path | str]
        - imgs: List[np.ndarray] (H,W,3) RGB
        - masks: List[np.ndarray] (H,W,2)
        - cls_ids: List[int]
        '''
        # check len with img_ids batch
        def verify_batch_items(img_ids, batch_item):
            if len(img_ids) != len(batch_item):
                for img_id, batch_item in zip(img_ids, batch_item):
                    print(f"img_id:  {img_id}")
                    print(f"type batch_item: {type(batch_item)}")
                raise ValueError("Length of item in next batch does not match length of img_ids")
            


        fail_img_ids = [] #img id for unqualify img files
        got_fail = False


        file_names, imgs, masks, cls_ids = [], [], [], []

        # batch should be based on img_ids, as all the filter logic will done on image_df
        img_ids = self.get_batch_img_ids(n, by="image")
        anns = self.get_anns_by_img_ids(img_ids)

        for ann in anns:
            img_id = ann['image_id']
            img_record = self.coco.imgs.get(img_id, None)
            if img_record:

                # read img
                fname = img_record['file_name']
                fpath = self.img_root_dir/fname
                if not fpath.is_file():
                    print(f"file <{fname}> not found skip requested image")
                    fail_img_ids.append(img_id)
                    got_fail = True
                    continue

                try:
                    img = read_color_img(fpath)
                except ValueError as e:
                    if 'Only color img allowed' in e.args:
                        print('skip non color img')
                        fail_img_ids.append(img_id)
                        got_fail = True
                        continue
                    else:
                        raise e
                
                # decode mask
                mask = self.coco.annToMask(ann)
                if mask is None:
                    print(f"segmentation not found for ann_id: <{ann['id']}>")
                    continue
                
                imgs.append(img)
                masks.append(mask)
                file_names.append(fname)
                cls_ids.append(ann['category_id'])
            else:
                print(f'image_id: <{img_id}>not found')
                continue
        
        img_ids = pd.Series(img_ids)
        succes_img_ids = img_ids[~img_ids.isin(fail_img_ids)]

        if got_fail:
            print(f"img_ids:  {list(img_ids)}")
            print(f"succes_img_ids:  {list(succes_img_ids)}")
            print(f"file_names:  {file_names}")
            print(f"imgs:  {imgs}")
            print(f"masks:  {masks}")
            print(f"cls_ids:  {cls_ids}")
            dash()

        for batch_item in [file_names, imgs, masks, cls_ids]:
            verify_batch_items(succes_img_ids, batch_item)

        if create_obj:
            masked_imgs = [MaskedImage(img, mask, cls_id) for img, mask, cls_id in zip(imgs, masks, cls_ids)]
            return masked_imgs
        else:
            return file_names, imgs, masks, cls_ids

class LoadWeightedCoCoImage(LoadCocoImage):
    WEIGHT_SUFFIX = '--weight'

    def __init__(self, coco_path, img_root_dir=None):
        super().__init__(coco_path, img_root_dir=img_root_dir)
        self.has_sample_weight = False
        self.set_weighted_img_df()

    def set_weighted_img_df(self):
        self.weighted_img_df = self.coco_df['images'].copy()

    def set_sample_weight(self):
        '''merge weight based on cols with WEIGHT_SUFFIX as suffix '''
        self.weighted_img_df['sample_weight'] = 1
        weight_cols = [c for c in self.weighted_img_df.columns if c.endswith(self.WEIGHT_SUFFIX)]
        for c in weight_cols:
            self.weighted_img_df['sample_weight'] *= self.weighted_img_df[c]
        self.has_sample_weight = True

    def get_batch_img_ids(self, n, by="image"):
        options = ["ann", "image"]

        if by == "ann":
            return self.anns_df["image_id"].sample(n)
        elif by == "image":
            if self.has_sample_weight:
                return self.weighted_img_df.sample(
                    n, weights='sample_weight')['id']
            else:
                return self.coco_df["images"]["id"].sample(n)
        else:
            raise ValueError(f"only {options} allowed")