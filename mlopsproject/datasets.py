import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CellsDataset(Dataset):
    def __init__(self, data_path, mask_path, aug=None):
        """
        Args:
            data_path: путь до изображений.
            mask_path: путь до масок изображений.
        """
        self.data_path = data_path
        self.mask_path = mask_path

        # Файлы с изображениями и масками
        self.files = os.listdir(data_path)
        self.mask_files = os.listdir(self.mask_path)

        assert len(self.files) == len(self.mask_files)

        # Сортируем файлы, чтобы было соответствие
        # между изображениями и масками за счет порядка следования.
        self.files.sort()
        self.mask_files.sort()

        # агументация из albumentations
        self.aug = aug

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Загружаем изображение и маску
        file_name = os.path.join(self.data_path, self.files[idx])
        mask_name = os.path.join(self.mask_path, self.mask_files[idx])

        # Пробразуем изображение и маску в numpy массив
        input = np.array(Image.open(file_name), dtype=np.uint8)
        target = np.array(Image.open(mask_name), dtype=np.uint8)

        # применяем аугментацию
        if self.aug:
            augmented = self.aug(image=input, mask=target)
            input = augmented["image"]
            target = augmented["mask"]

        # исправим порядок размерностей
        input = input.transpose(2, 0, 1)

        # Приводим к torch tensor
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.from_numpy(target)

        # Приводим таргет/маску к такому виду,
        # чтобы его значения были от 0 до 1
        target = (target > 0).int().unsqueeze(0)

        return input, target
