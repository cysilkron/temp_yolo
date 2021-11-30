from pathlib import Path
from itertools import chain
from pprint import pprint
import numpy as np
import pandas as pd
from collections import defaultdict
from IPython.display import display, display_html
from typing import List, Dict, Union


def all_len_is_same(*args):
    return all(len(args[0]) == len(_arg) for _arg in args[1:])

def dash(n=50):
	print('-'*n)

def ls(p, r='*'):
    return list(p.glob(r))

def lls(p, r='*'):
    return len(ls(p ,r))

def flat(to_chain: list) -> list:
    return list(chain(*to_chain))

def rename_dict(dict_, rename_keys: dict):
    for old_key, new_key in rename_keys.items():
        dict_[new_key] = dict_.pop(old_key)
    return dict_
    

def is_key_match(dict_1, dict_2):
    dict_1_keys = pd.Series(list(dict_1.keys()))
    dict_2_keys = pd.Series(list(dict_2.keys()))
    return dict_1_keys.isin(dict_2_keys).all()

def mkdir_r(dir_path, exist_ok=False):
    Path(dir_path).mkdir(parents=True, exist_ok=exist_ok)

def len_dict(dict_):
    return {k: len(list_) for k, list_ in dict_.items()}

def pd_len_dict(dict_list: List[dict]) -> pd.DataFrame:
    display(pd.DataFrame([len_dict(d) for d in dict_list]))
    
def arr_map(func, arr):
    return np.array(list(map(func, arr)))

#pd dataframe
def df_display_beside(dfs, names=[]):
    html_str = ''
    if names:
        html_str += ('<tr>' + 
                     ''.join(f'<td style="text-align:center">{name}</td>' for name in names) + 
                     '</tr>')
    html_str += ('<tr>' + 
                 ''.join(f'<td style="vertical-align:top"> {df.to_html(index=False)}</td>' 
                         for df in dfs) + 
                 '</tr>')
    html_str = f'<table>{html_str}</table>'
    html_str = html_str.replace('table','table style="display:inline"')
    display_html(html_str, raw=True)

def pd_display_max(row=20, col=10, show_all=False):
    '''
    if show_all, display all row and col,
    ignore the row and col setting
    '''
    
    if show_all:
        pd.set_option('display.max_rows', None, 'display.max_columns', None)
    else:
        pd.set_option('display.max_rows', row, 'display.max_columns', col)


Path.ls = ls
Path.lls = lls