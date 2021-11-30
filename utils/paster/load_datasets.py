from .load_coco import LoadWeightedCoCoImage
from typing import Dict, List
import re
import pandas as pd
import matplotlib.pyplot as plt
from utils.tools.general import flat
from utils.tools.plot import plt_bar_dist

def map_weight_dict(field, weight_dict):
    return weight_dict.get(field, 0)

def weighted_sample_values_count(num, df, field, weights_col, frac=0.5):
    def sample_counts(df, n, fraction=False):
        if fraction:
            return df.sample(
                frac=n,
                weights=weights_col)[field].value_counts().sort_index()
        else:
            return df.sample(
                n, weights=weights_col)[field].value_counts().sort_index()

    #if n_sample >= num: ??
    samples_dist = sample_counts(df, frac, fraction=True)
    n_sample = samples_dist.values.sum()
    if n_sample >= num:
        print(
            'warning: num of sample required is lower than fraction, sampled class might be imbalanced'
        )
        samples_dist = sample_counts(df, num, fraction=False)
    else:
        loop = (num // n_sample) - 1
        rmd = num % n_sample

        for _ in range(loop):
            samples_dist += sample_counts(df, frac, fraction=True)
        samples_dist += sample_counts(df, rmd, fraction=False)

    if np.isnan(samples_dist.values.sum()):
        print(f"rmd:  {rmd}")
        print(f"num:  {num}")

    print('total_sample: ', samples_dist.values.sum())
    return samples_dist

def values_count(df, field=None):
    if field:
        return df[field].value_counts().sort_index()

    return df.value_counts().sort_index()

def map_weight_dict(field, weight_dict):
    return weight_dict.get(field, 0)


def weighted_sample_value_counts(total, bs, df, field, weights_col):
    print(f"total:  {total}")
    print(f"bs:  {bs}")

    def sampling(bs):
        return df.sample(bs, weights=weights_col)[field]

    if bs > total:
        raise ValueError('batch size cant be smaller than total samples')

    rmd = total % bs
    rmd_batch = 1 if rmd > 0 else 0
    batches = (total // bs) - 1 + rmd_batch

    sample_df = sampling(bs)
    for i in range(batches - 1):
        sample_df = pd.concat([sample_df, sampling(bs)])

    last_bs = rmd if rmd_batch else bs
    sample_df = pd.concat([sample_df, sampling(last_bs)])

    print(f"len(sample_df):  {len(sample_df)}")

    return sample_df.value_counts().sort_index()


def balance_weight(values_count: pd.Series, sigma=1):
    '''balance the img to be chosen by inversing the distribution of categories occurance'''
    def mean_normalize(a):
        return (a - a.mean()) / a.std()

    def min_max_normalize(a):
        return (a - a.min()) / (a.max() - a.min())

    def normalize(a):
        try:
            normed_a = min_max_normalize(a)

        except ZeroDivisionError:
            normed_a = mean_normalize(a)
            normed_a = normed_a + abs(normed_a.min())

        return normed_a

    inverse_deviation = -(values_count.values - values_count.values.mean())
    #     max_deviation = cat_dist.values - cat_dist.values.max()
    return dict(zip(values_count.index, 
                normalize(inverse_deviation) + sigma))


class LoadWeightedCategories(LoadWeightedCoCoImage):
    def __init__(self, coco_path, img_root_dir=None):
        super().__init__(coco_path, img_root_dir=img_root_dir)

    def set_cat_weights(self, sigma=2, cat_ids:List = None, excluded_cat_id_weight: dict= None, plot=False):
            '''
            set the category_ids weight with class balancing, the excluded class weight can be explicitly map with dictionary `excluded_cat_id_weight`
            Args:
                cat_ids: List[int] (optional)
                    the list of category id to be included for class balancing, if not pass, all class id will be chosen by default
                excluded_cat_id_weight : dict[int, int] (optional)
                    Eg: {-1: 2, 14: 1}
            '''


            cat_and_img_df = pd.merge(self.coco_df['images'], self.coco_df['annotations'], left_on='id',
                        right_on='image_id')[['category_id', 'image_id']]

            if cat_ids:
                if not excluded_cat_id_weight:
                    raise ValueError('`excluded_cat_id_weight` must be set if the `cat_ids` is provided')
                is_cat = cat_and_img_df['category_id'].isin(cat_ids)
                cat_dist = cat_and_img_df[is_cat]['category_id'].value_counts().sort_index()
            else:
                cat_dist = cat_and_img_df['category_id'].value_counts().sort_index()


            cat_weight = balance_weight(cat_dist, sigma=sigma)
            if excluded_cat_id_weight:
                for k,v in excluded_cat_id_weight.items():
                    if not (isinstance(v, int) or isinstance(v, float)):
                        raise TypeError('weight bust be number')
                cat_weight = {**cat_weight, **excluded_cat_id_weight}

            if plot:
                plt_bar_dist(cat_weight, 'cat weight')

            self.weighted_img_df['cat'+self.WEIGHT_SUFFIX] = cat_and_img_df['category_id'].apply(
                lambda x: map_weight_dict(x, cat_weight))

            self.weighted_img_df['category_id'] = cat_and_img_df['category_id']

    def plot_category_sample(self, bs:int = 4, loop_n:int = 1000, title=None):

        batch_img_ids = flat([self.get_batch_img_ids(bs, by='image') for i in range(loop_n)])
        test_anns = self.get_anns_by_img_ids(batch_img_ids)
        cat_ids = pd.DataFrame(test_anns)['category_id']
        plt_bar_dist(cat_ids.value_counts().sort_index(), title)

class LoadProduct(LoadWeightedCategories):

    def __init__(self, coco_path, img_root_dir=None):
        super().__init__(coco_path, img_root_dir=img_root_dir)


        #example flow
        # lb_coco.filter_front_img(corrupted_imgs)
        # lb_coco.set_weighted_img_df()
        # lb_coco.set_angle_weights({
        #             't45': 1,
        #             'tb45': 1,
        #             't90': 0,
        #             'tb90': 0,
        #             'h45': 1,
        #             'hb45': 1,
        #             'not_found': 1
        #         })
        # lb_coco.plot_category_sample(3, 3000, title='before set cat weight')
        # lb_coco.set_cat_weights(sigma=1)
        # lb_coco.set_sample_weight()

    def set_angle_weights(self, angle_weights: Dict):
        def find_angle(fname):
            angle_str = re.findall(r'(\w+)(?:__)', fname)
            if len(angle_str):
                return angle_str[0]
            return 'not_found'

        self.weighted_img_df['angles'] = self.weighted_img_df[
            'file_name'].apply(find_angle)
        print(' angles list:', self.weighted_img_df['angles'].unique())

        # angle_weights = {
        #     't45': 1,
        #     'tb45': 1,
        #     't90': 0,
        #     'tb90': 0,
        #     'h45': 1,
        #     'hb45': 1,
        #     'not_found': 1
        # }

        self.weighted_img_df['angles'+self.WEIGHT_SUFFIX] = self.weighted_img_df[
            'angles'].apply(lambda x: map_weight_dict(x, angle_weights))

class LoadNoise(LoadWeightedCategories):
    def __init__(self, coco_path, img_root_dir=None):
        super().__init__(coco_path, img_root_dir=img_root_dir)
        # self.set_coco(*args, **kwargs)
        self.coco.anns = self.map_anns_to_noise(self.coco.anns)
        self.coco.cats = self.map_cats_to_noise(self.coco.cats)
        self.set_coco_df()
        self.set_front_fnames(img_root_dir)
        self.filter_front_img([])
        self.has_sample_weight = False
        self.set_weighted_img_df()

    def map_catId_2_noise(self, df, key):
        df[key] += 1
        df[key] = -df[key]
        return df

    def map_anns_to_noise(self, anns):
        df = pd.DataFrame(anns).T
        df = self.map_catId_2_noise(df, 'category_id')
        return df.T.to_dict()

    def map_cats_to_noise(self, cats):
        df = pd.DataFrame(cats).T
        df = self.map_catId_2_noise(df, 'id')
        return df.T.to_dict()
    
