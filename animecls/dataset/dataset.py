'''Dataset class'''

from __future__ import annotations
from typing import Callable

from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

from animecls.dataset.utils import prepare_labels

class ImageFolder(Dataset):
    '''ImageFolder implementation which expects a list object of files instead a path.

    Arguments:
        file_list: list[str]
            List of paths to image files.
        folder_list: list[str]
            List of folders which represents the class of the images.
        meta_root: str
            Path to the folder which contains 'folder2label.pickle' and 'label2name.pickle'.
        transform: Callable
            Function to transform the input image.
        target_transform: Callable (default: lambda x:x)
            Function to transform the target.
    '''
    def __init__(self,
        file_list: list[str], folder_list: list[str], meta_root: str, transform: Callable, target_transform: Callable=lambda x:x
    ) -> None:
        super().__init__()
        self.files = file_list
        labels, label2name = prepare_labels(folder_list, meta_root)
        self.labels = labels
        self.label2name = label2name

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        label = self.labels[index]

        image = Image.open(file).convert('RGB')
        image = self.transform(image)
        label = self.target_transform(label)

        return image, label
