
from .general import *
from .bbox import xywh2xyxy, xyxy2xywh
import uuid

import imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.augmenters as iaa
import yaml


#read files
def yoloLabel_lines_2_list(img, yolo_lines):
    ih, iw , _ = img.shape
    yolo_lines = np.array(yolo_lines).astype('float')
    bboxes = yolo_lines[:, 1:5]
    bboxes = bboxes * [iw, ih, iw, ih]
    bboxes = xywh2xyxy(bboxes)
    yolo_lines[:, 1:5] = bboxes

    return yolo_lines.astype('int')

def f_readlines(fname, split_char=None):
    with open(str(fname), 'r') as f:
        lines = f.read().splitlines()

    if split_char:
        return np.array([line.split(split_char) for line in lines])

    return lines

def f_readlabels(img, fname):
    yolo_lines = f_readlines(fname, split_char=' ')
    labels = yoloLabel_lines_2_list(img, yolo_lines)

    return labels

def lbl_from_img_name(lbl_dest, img_fname):
    fname = Path(img_fname).stem
    lbl_path = lbl_dest/(fname + '.txt')
    return lbl_path


# write labels
def uniq_file_name(prefix=None):
    hex_name = uuid.uuid4().hex
    uniq_name = prefix + '_' + hex_name if prefix else  hex_name
    return uniq_name

def list_2_yoloLabel_lines(img, labels_list):
    ih, iw , _ = img.shape
    labels_list = np.array(labels_list).astype('float')
    
    bboxes = labels_list[:, 1:5] 
    bboxes  = (bboxes / [iw, ih, iw, ih]).round(4)
    bboxes = xyxy2xywh(bboxes)
    labels_list[:, 1:5] = bboxes

    return labels_list.astype('str') #str lines


def f_writelines(lines: List[str], fname, join_by=None):
    if join_by is None:
        lines = [str(line) + '\n' for line in lines]
    else:
        lines = [join_by.join(list(line)) +'\n' for line in lines]

    with open(str(fname), 'w') as f:
        f.writelines(lines)

def write_img_and_bboxes(img, labels, img_dest, lbl_dest):
    fname = uniq_file_name('img')
    img_name = fname + '.jpg'
    lbl_name = fname + '.txt'
    
    imageio.imwrite(img_dest/img_name, img)
    yolo_lines = list_2_yoloLabel_lines(img, labels)
    f_writelines(yolo_lines, lbl_dest/lbl_name, join_by=' ')


def img_list_from(img_path, img_file_type):
    """
    Eg:
        img_file_type = ['.jpg', '.png', '.jpeg']
    """

    def from_recursive_dir(img_path):
        print(f'Search for img in dir...')
        return [p for p in tqdm(img_path.rglob('*'))
                if p.suffix in img_file_type]

    def from_paths_in_txt(img_path):
        return f_readlines(img_path)

    if img_path.is_dir():
        img_list = from_recursive_dir(img_path)

    elif img_path.is_file() and img_path.suffix == '.txt':
        img_list = from_paths_in_txt(img_path)

    else:
        raise ValueError(
            'only list of Path and directory of imgs is supported')

    return img_list


def read_color_imgs(img_paths: pd.Series, n=int, verbose=True):
    img_to_read = img_paths.sample(n)
    for p in img_to_read:
        img = imageio.imread(p)
        if img.ndim == 3:
            yield img
        else:
            if verbose:
                print('pass grey scale img')
            yield next(read_color_imgs(img_paths, 1))


# yaml
def dump_yaml(dict_, fpath):
    with open(str(fpath), 'w') as f:
        yaml.dump(dict_, f, sort_keys=False)

def load_yaml(fpath):
    with open(str(fpath)) as f:
        dict_ = yaml.load(f, Loader=yaml.FullLoader)
        return dict_