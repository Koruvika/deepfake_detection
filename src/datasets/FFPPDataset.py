import torch 
import os 
from torch.utils.data import Dataset
from PIL import Image
import cv2

def one_hot_encoding(label:int, num_classes=2):
    vec = torch.zeros(size=(num_classes,), dtype=torch.float32)
    vec[label] = 1.
    return vec

class FFPPDataset(Dataset):
    def __init__(self, dataset_configs, transform=None) -> None:
        super(FFPPDataset, self).__init__()
        self.transform = transform 
        self.config = dataset_configs

        self.image_label = [(image_fp, 1) \
                        for deepfake_folder in self.config.deepfake_folders \
                        for image_folder in os.listdir(deepfake_folder) \
                        for image_fp in os.listdir(os.path.join(deepfake_folder, image_folder))]
        self.image_label = self.image_label + [(image_fp, 0)\
                        for original_folder in self.config.original_folders \
                        for image_folder in os.listdir(original_folder) \
                        for image_fp in os.listdir(os.path.join(original_folder, image_folder))]
        
        self.image_fps, self.label = zip(*self.image_label)

    def __len__(self):
        return len(self.image_fps)
    
    def __getitem__(self, item):
        image_fp = self.image_fps[item]
        image = cv2.imread(image_fp, cv2.IMREAD_UNCHANGED)

        label = self.label[item]

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        label = one_hot_encoding(label, 2)
        return image, label 


class CelebValidateDataset(Dataset):
    def __init__(self, dataset_configs, phase="test", transform=None):
        self.configs = dataset_configs
        self.phase = phase
        self.transform = transform

        self.images_fp = []
        self.label = []

        video_list1 = os.listdir(os.path.join(self.configs.test_root, "Celeb-real"))
        video_list1 = [os.path.join(self.configs.test_root, "Celeb-real", v) for v in video_list1]
        for video in video_list1:
            im_fn = os.listdir(video)
            im_fp = [os.path.join(video, fn) for fn in im_fn]
            self.images_fp = self.images_fp + im_fp
            self.label = self.label + [1 for _ in range(len(im_fn))]

        video_list1 = os.listdir(os.path.join(self.configs.test_root, "YouTube-real"))
        video_list1 = [os.path.join(self.configs.test_root, "YouTube-real", v) for v in video_list1]
        for video in video_list1:
            im_fn = os.listdir(video)
            im_fp = [os.path.join(video, fn) for fn in im_fn]
            self.images_fp = self.images_fp + im_fp
            self.label = self.label + [1 for _ in range(len(im_fn))]

        video_list1 = os.listdir(os.path.join(self.configs.test_root, "Celeb-synthesis"))
        video_list1 = [os.path.join(self.configs.test_root, "Celeb-synthesis", v) for v in video_list1]
        for video in video_list1:
            im_fn = os.listdir(video)
            im_fp = [os.path.join(video, fn) for fn in im_fn]
            self.images_fp = self.images_fp + im_fp
            self.label = self.label + [0 for _ in range(len(im_fn))]

    def __len__(self):
        return len(self.images_fp)

    def __getitem__(self, item):
        image_fp = self.images_fp[item]
        image = cv2.imread(image_fp, cv2.IMREAD_UNCHANGED)

        label = self.label[item]

        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        label = one_hot_encoding(label, 2)
        return image, label