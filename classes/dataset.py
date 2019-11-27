import copy
import numpy as np
import os
from torch.utils.data import Dataset
import pandas as pd
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import cv2
import re


class ImageDataset(Dataset):
    def __init__(self, root_dir, idxs=None, csv_file=None, transform=None, oversample=False):
        self.oversample = oversample
        self.labels_init = pd.read_csv(csv_file, index_col=0) if csv_file else None

        if self.labels_init is not None:
            assert idxs is not None, "If csv_file is passed, idxs must be passed as well."
            self.labels_init = self.labels_init.loc[idxs, :]  # Original copy of labels
            self.labels = copy.deepcopy(self.labels_init)  # Labels actually used (can be augmented via oversampling
            self.idxs = list(idxs)
        else:
            self.labels = None
            self.idxs = None

        self.root_dir = root_dir
        self.transform = transform

        if self.oversample:
            self.reapply_oversample()

        # Preset length
        if self.labels is not None:
            self.length = len(self.labels)
        else:
            image_names = os.listdir(self.root_dir)
            r = re.compile('\d+')
            self.idxs = sorted([int(r.search(x)[0]) for x in image_names])
            self.length = len(image_names)

        # Store output image size
        sample_output = self.__getitem__(0)
        self.num_channels, self.img_dim1, self.img_dim2 = sample_output['image'].shape

        # Hardcoded for simplicity
        self.num_classes = 3

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.labels is not None:
            img_name = os.path.join(self.root_dir,
                                    self.labels.iloc[idx, 0] + '.png')
        else:
            img_name = os.path.join(self.root_dir,
                                    'test_' + str(self.idxs[idx]) + '.png')

        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = torch.from_numpy(image)

        if self.labels is not None:
            label = self.labels.iloc[idx, 1:]
            label = torch.Tensor(label)

            sample = {
                'image': image,
                'label': label,
                'IDs': self.idxs[idx]
            }

            return sample

        else:
            return {
                'image': image,
                'IDs': self.idxs[idx]
            }

    def reapply_oversample(self, threshold=5):
        """
        Creates a replica of self.labels that balanced the occurrence of each labels. Seeks to increase the number of each label to be the same frequency as the
        most frequently occurring label.
        """
        self.labels = copy.deepcopy(self.labels_init)

        # While not all values are equal to the maximum value
        while not (abs(self.labels.iloc[:, 1:].values.sum(0) - self.labels.iloc[:, 1:].values.sum(0).max()) <= threshold).all():
            # Determine classes to target in this iteration of the loop
            target_classes = np.argwhere(~(abs(self.labels.iloc[:, 1:].values.sum(0) - self.labels.iloc[:, 1:].values.sum(0).max()) <= threshold))

            # For each target class, randomly sample one image including that class and append
            for target_class in target_classes:
                self.labels = self.labels.append(self.labels_init[self.labels_init.iloc[:, 1 + target_class.item()] > 0].sample(1))

        self.length = len(self.labels)
        r = re.compile('\d+')
        self.idxs = [int(r.search(x)[0]) for x in self.labels.ID]

    def collect_examples(self, label, num_examples):
        """
        Retrieves a specified number of examples with a specified label
        :param label: List of length 3 corresponding to three possible labels
        :param num_examples: Maximum number of examples to return. Will retrieve minimum of this value and the number of examples available
        :return: 4D Tensor
        """
        assert self.labels is not None, "Cannot collect examples without labels."

        examples = self.labels.index[(self.labels.iloc[:, 1:] == np.array(label)).all(1)]
        num_examples = min(num_examples, len(examples))

        output_tensor = torch.Tensor()
        for i in range(num_examples):
            output_tensor = torch.cat((output_tensor, self.__getitem__(examples[i])['image'].view(1, self.num_channels, self.img_dim1, self.img_dim2)), dim=0)

        return output_tensor

    def show_grid(self, label, num_examples):
        tensors = self.collect_examples(label=label, num_examples=num_examples)
        grid = vutils.make_grid(tensors, nrow=int(np.ceil(num_examples ** 0.5)), normalize=True)
        title_list = [col + ': ' + ('True' if label_bool else 'False') for label_bool, col in zip(label, self.labels.columns[1:])]

        plt.figure(figsize=(14, 14))
        plt.axis('off')
        plt.title(', '.join(title_list))
        plt.imshow(np.transpose(grid, (1, 2, 0)), cmap=plt.cm.bone)
        plt.show()

    def show_image(self, idx):
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(np.transpose(self.__getitem__(idx)['image'], (1, 2, 0)), cmap=plt.cm.bone)
        plt.title(f'Image ID: {idx}')
