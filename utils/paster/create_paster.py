from utils.paster.paster import PasteOnLabeledImage
from utils.paster.create_datasets import LoadProductWithNoise


def create_train_paster():
    mixed_coco = LoadProductWithNoise()
    return PasteOnLabeledImage(mixed_coco, debug=True, save_mask=True)
