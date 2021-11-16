import pandas as pd
import pickle
import random
from collections import namedtuple
from typing import Tuple

import cv2
import lmdb
import numpy as np
from path import Path

Sample = namedtuple('Sample', 'gt_text, file_path')
Batch = namedtuple('Batch', 'imgs, gt_texts, batch_size')


class DataLoaderIAM:
    """
    Loads data which corresponds to IAM format,
    see: http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
    """

    def __init__(self,
                 train_img,
                 train_csv,
                 validate_img,
                 validate_csv,
                 batch_size: int,
                 data_split: float = 0.95):
        """Loader for dataset."""

        self.data_augmentation = False
        self.curr_idx = 0
        self.batch_size = batch_size
        self.samples = []
        self.train_samples = []
        self.validation_samples = []

        ## self code
        base_path_train = './dataset/'
        base_path_validate = './dataset/'

        chars = set()
        # loading test images
        file = pd.read_csv(base_path_train + train_csv)
        image_name_list = list(file['Image'])
        gt_list = list(file['Word'])
        for i in range(len(image_name_list)):
            chars = chars.union(set(list(gt_list[i])))
            self.train_samples.append(Sample(gt_list[i], base_path_train + train_img + image_name_list[i]))

        #loading validation images
        file = pd.read_csv(base_path_validate + validate_csv)
        image_name_list = list(file['Image'])
        gt_list = list(file['Word'])
        for i in range(len(image_name_list)):
            chars = chars.union(set(list(gt_list[i])))
            self.validation_samples.append(Sample(gt_list[i], base_path_validate + validate_img + image_name_list[i]))

        # put words into lists
        self.train_words = [x.gt_text for x in self.train_samples]
        self.validation_words = [x.gt_text for x in self.validation_samples]

        # start with train set
        self.train_set()

        # list of all chars in dataset
        self.char_list = sorted(list(chars))

    def train_set(self) -> None:
        """Switch to randomly chosen subset of training set."""
        self.data_augmentation = True
        self.curr_idx = 0
        random.shuffle(self.train_samples)
        self.samples = self.train_samples
        self.curr_set = 'train'

    def validation_set(self) -> None:
        """Switch to validation set."""
        self.data_augmentation = False
        self.curr_idx = 0
        self.samples = self.validation_samples
        self.curr_set = 'val'

    def get_iterator_info(self) -> Tuple[int, int]:
        """Current batch index and overall number of batches."""
        if self.curr_set == 'train':
            num_batches = int(np.floor(len(self.samples) / self.batch_size))  # train set: only full-sized batches
        else:
            num_batches = int(np.ceil(len(self.samples) / self.batch_size))  # val set: allow last batch to be smaller
        curr_batch = self.curr_idx // self.batch_size + 1
        return curr_batch, num_batches

    def has_next(self) -> bool:
        """Is there a next element?"""
        if self.curr_set == 'train':
            return self.curr_idx + self.batch_size <= len(self.samples)  # train set: only full-sized batches
        else:
            return self.curr_idx < len(self.samples)  # val set: allow last batch to be smaller

    def _get_img(self, i: int) -> np.ndarray:
        img = cv2.imread(self.samples[i].file_path, cv2.IMREAD_GRAYSCALE)
        return img

    def get_next(self) -> Batch:
        """Get next element."""
        batch_range = range(self.curr_idx, min(self.curr_idx + self.batch_size, len(self.samples)))

        imgs = [self._get_img(i) for i in batch_range]
        gt_texts = [self.samples[i].gt_text for i in batch_range]


        print(gt_texts[0])
        import matplotlib.pyplot as plt
        plt.imshow(imgs[0])
        plt.show()
        
        self.curr_idx += self.batch_size
        return Batch(imgs, gt_texts, len(imgs))
