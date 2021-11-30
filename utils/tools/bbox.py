import torch
import numpy as np

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def convert_labels(labels, from_bbox: str, to_bbox:str):
    '''
    labels: np.ndarray | torch.tensor
        labels should be in (Nx4) format :
            [[class_id, *bbox]]

    from_bbox: str {'xyxy', 'xywh', 'xywh_normed'}
    to_bbox: str {'xyxy', 'xywh', 'xywh_normed'}
        Determines the type of the bbox and will be transform into 2 dimension array with
        the format [[cls_id, x1, y1, x2, y2]]
    
        format allowed:

            'xyxy' (default) : array like [[class_id, x1, y1, x2, y2]] with dtype (int)

            'xywh' : array like [[class_id, x, y  width, height]] with dtype (int)

            'xywh_normed' : array like [[class_id, x, y, width, height]] (normalized) with dtype (float32)
        '''

    label_formats = ['xyxy',  'xywh', 'xywh_normed']

    if from_bbox not in label_formats or to_bbox not in label_formats:
        raise ValueError(f'Invalid label_foramt, only these format: `{label_formats}` is supported')

    if from_bbox == 'xywh' and to_bbox == 'xyxy':
        labels[:, 1:] = xywh2xyxy(labels[:, 1:])

    elif label_format == 'xywh_normed' and to_bbox == 'xyxy':
        labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h)

    elif from_bbox == 'xyxy' and to_bbox == 'xywh':
        labels[:, 1:] = xywh2xyxy(labels[:, 1:])

    elif label_format == 'xyxy' and to_bbox == 'xywh_normed':
        labels[:, 1:] = xyxy2xywhn(labels[:, 1:], w, h)

    return labels


def yolo2xyxy(bbox):
    x_mid,y_mid, w,h = bbox
    
    x1 = x_mid - (w/2)
    y1 = y_mid - (h/2)
    x2 = x_mid + (w/2)
    y2 = y_mid + (h/2)
    
    return [x1, y1, x2, y2]


def xyxy2yolo(bbox):
    x1, y1, x2, y2 = bbox
    
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    
    x_mid = x1 + (w/2)
    y_mid = y1 + (h/2)
    
    return [x_mid, y_mid, w, h]