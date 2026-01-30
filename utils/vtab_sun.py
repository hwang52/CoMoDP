from itertools import count
import torch
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from typing import Any, Tuple, Callable, Optional

import PIL.Image

# from torchvision.datasets import SUN397
from torchvision.datasets.vision import VisionDataset


def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
    print('==> Transition Matrix:')
    print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def binarize_class(y):  
    label = y.reshape(len(y), -1)
    enc = OneHotEncoder(categories='auto') 
    enc.fit(label)
    label = enc.transform(label).toarray().astype(np.float32)     
    label = torch.from_numpy(label)
    return label


class MySUN397(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.

    Args:
        root (string): Root directory of the dataset.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"

    def __init__(
        self,
        root: str,
        w_transform: Optional[Callable] = None,
        s_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        download: bool = False,
        data_tuple: tuple = None,
        est: bool = False,
        partial_rate: int = 0
    ) -> None:
        super().__init__(root, transform=None, target_transform=target_transform)
        self._data_dir = Path(self.root) / "SUN397"

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]
        self.num_class = len(self.classes)
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        self._est = est

        if data_tuple is None:
            self._image_files = list(self._data_dir.rglob("sun_*.jpg"))
            self._image_files = np.array(self._image_files)

            self._labels = [
                self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])] for path in self._image_files
            ]
            self._labels = np.array(self._labels)
            counts = np.array([(self._labels == i).sum() for i in range(self.num_class)])
            labelmap = {(- counts).argsort()[i]:i for i in range(self.num_class)}
            self._labels = [
                int(labelmap[label]) for label in self._labels
            ]
            # rearrange labels according to sorted counts
            self._labels = np.array(self._labels)

            # split train_test
            test_idx = self.__split_train_test__()
            self.image_files_test = self._image_files[test_idx]
            self.labels_test = self._labels[test_idx]
            self._image_files = self._image_files[~test_idx]
            self._labels = self._labels[~test_idx]

            # generate candidate labels
            self._labels = torch.from_numpy(self._labels)
            self._train_labels = generate_uniform_cv_candidate_labels(self._labels, partial_rate)

            self.train_tuple = (self._image_files, self._labels)
            self.test_tuple = (self.image_files_test, self.labels_test)

            self.w_transform = w_transform
            self.s_transform = s_transform
            self._train = True
        else:
            (self._image_files, self._labels) = data_tuple
            self._train_labels = binarize_class(self._labels).float() if est else self._labels
            self.test_transform = test_transform
            self._train = self._est
            # if est mode, set train=True

    def get_test_data(self, train=False):
        return self.train_tuple if train else self.test_tuple

    def __split_train_test__(self):
        labels = np.array(self._labels)
        test_idx = np.zeros(len(labels)) > 1
        
        # All false array
        classes = np.arange(self.num_class)
        img_num_per_cls = np.ones(self.num_class).astype(int) * 50
        # 50 examples per class
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            idx = np.where(labels == the_class)[0]
            np.random.shuffle(idx)
            test_idx[idx[:the_img_num]] = True
        return test_idx

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        image_file = self._image_files[idx]
        label = self._train_labels[idx]

        image = PIL.Image.open(image_file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        if self._train:
            true_label = self._labels[idx]
            if not self._est:
                image_w = self.w_transform(image)
                image_s = self.s_transform(image)
                return image_w, image_s, label, true_label, idx
            else:
                return self.test_transform(image), label, true_label
        else:
            image = self.test_transform(image)
            return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        # download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)

    def __len__(self) -> int:
        return len(self._image_files)