import cv2

from dataset.load_data import get_train_and_val_data
from utils.image_aug import aug_img_func


if __name__ == '__main__':
    val_image_path_list, val_class_id, train_image_path_list, train_class_id = get_train_and_val_data()

