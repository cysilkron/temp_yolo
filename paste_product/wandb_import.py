#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pathlib import Path
import sys

yolo_dir = Path.home()/'yolov5'
if yolo_dir.is_dir():
    sys.path.append(str(yolo_dir))


# In[4]:


from typing import List
from utils.tools.general import *
import utils.tools.file as fu

from typing import List


# In[5]:


kaggle = Path.home()/'kaggle'
in_path = kaggle/'input'
out_path = kaggle/'working'

def mkdir_ignore(dir_path):
    dir_path.mkdir(parents=True, exist_ok=True)

mkdir_ignore(in_path)
mkdir_ignore(out_path)


# In[6]:


def format_size(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'Kb', 2: 'Mb', 3: 'Gb', 4: 'Tb'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f}{power_labels[n]}"
#     return size, 


# In[ ]:


import wandb


# In[14]:


SEED=1234


# # verify filepaths with dataframe

# In[8]:


def create_path_df(file_paths: List[Path]):
    parent, stem, suffix, file_path, is_file = np.array([(p.parent, p.stem, p.suffix, p, p.is_file())
                                                             for p in tqdm(file_paths)]).T
    df = pd.DataFrame({
        'parent': parent,
        'stem': stem,
        'suffix': suffix,
        'file_path': file_path,
        'is_file': is_file
    })
    
    return df


# In[10]:


def split_col_name_from_suffix(col: str, join_str):
    tokens = col.rsplit(join_str, 1)
        
    if len(tokens) == 2:
        col_name, suffix = tokens
    elif len(tokens) == 1:
        col_name = tokens[0]
        suffix = None
    else:
        raise ValueError('unable to split the col_name correctly')
        
    return col_name, suffix

def join_img_and_lbl_df(img_df, lbl_df, merge_key='stem', display_cols: Tuple[str]=('file_path', 'suffix'),
                          join_str='-', suffixes: Tuple[str]=('img', 'lbl')):
    '''join img_df and lbl_df then display the essential cols only (if requested)
    Args:
    -----
    join_str
    '''
    
    suffixes = [join_str + s for s in suffixes]
    img_and_lbl_df = pd.merge(img_df, lbl_df, left_on=merge_key, right_on=merge_key,
                              suffixes=suffixes)
    
    columns_to_show = [merge_key]
    for fullname in img_and_lbl_df.columns:
        name, suffix =  split_col_name_from_suffix(fullname, join_str)
        if name in display_cols:
            columns_to_show.append(fullname)
    
    return img_and_lbl_df, columns_to_show


# # wandb create artifact with paths dataframe

# In[13]:





def create_artifact_dataset(img_and_lbl_df, name='dataset', artifact_type='raw_dataset', log_table=False):
    '''create artifact with list of lbl_paths 
    label_path will be capture
    
    todo:
        find how to log table with wandbImage without using the LoadImagesAndLabels class
    '''
    artifact = wandb.Artifact(name=name, type=artifact_type)
    
    
    for img_path in tqdm(img_and_lbl_df['file_path-img'], desc='add image files'):
        artifact.add_file(img_path, name='data/images/' + Path(img_path).name)
    
    for lbl_path in tqdm(img_and_lbl_df['file_path-lbl'], desc='add label files'):
        artifact.add_file(lbl_path, name='data/labels/' + Path(lbl_path).name)
 
    return artifact


# # split

# ## by dataframe

# In[15]:


def split_by_df(df, test_size=0.2, random_state=1234):
    '''
     Parameters
    ----------
    test_size : float or int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples.
    '''
    test_df = df.sample(frac=test_size) if isinstance(test_size, float) else df.sample(test_size)
    train_df = df[~df.index.isin(test_df.index)]
    
    return train_df, test_df

