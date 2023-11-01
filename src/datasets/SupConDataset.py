import os.path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from src.utils import RandomDownScale, random_get_hull, IoUfrom2bboxes, crop_face, dynamic_blend

# TODO 1: Crop Face
# TODO 2: Write SupConDataset
# TODO 3: Training without Deepfake Augmentation

data_csv = {
    "original": "original.csv",
    "DeepFakeDetection": "DeepFakeDetection.csv",
    "Deepfakes": "Deepfakes.csv",
    "Face2Face": "Face2Face.csv",
    "FaceShifter": "FaceShifter.csv",
    "FaceSwap": "FaceSwap.csv",
    "NeuralTextures": "NeuralTextures.csv",
}
data_csv = [os.path.join("/mnt/data/duongdhk/datasets/processed_deepfake_detection_dataset/FFPP", fp) for fp in data_csv.values()]


default_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class SupConDataset(Dataset):
    def __init__(self, dataset_config, phase="train", contrast_transform=None, use_contrast=True):
        super(SupConDataset, self).__init__()

        assert phase in ["train", "test", "val"]

        self.config = dataset_config
        self.phase = phase
        self.filelist = pd.DataFrame(columns=["Image Path", "Landmarks Path", "Video", "Label", "Type"])

        # contrastive transforms
        self.contrast_transform = contrast_transform
        if self.contrast_transform is None:
            self.contrast_transform = default_transform
        if use_contrast:
            self.contrast_transform = TwoCropTransform(self.contrast_transform)

        # read file list
        for csv_fp in data_csv:
            self.filelist = pd.concat([self.filelist, pd.read_csv(csv_fp)], ignore_index=True)
        self.filelist = self.filelist.sample(frac=1).reset_index()
        self.image_list = np.array(self.filelist["Image Path"])
        self.landmarks_list = np.array(self.filelist["Landmarks Path"])
        self.video_list = np.array(self.filelist["Video"])
        self.label_list = np.array(self.filelist["Label"])
        self.type_list = np.array(self.filelist["Type"])

        print(f"[Dataset] Init successfully!!")

    def assign_transform(self, transform):
        self.contrast_transform = TwoCropTransform(transform)

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):

        # read image, landmarks and label
        image = cv2.imread(self.image_list[item], cv2.IMREAD_UNCHANGED)
        im = image
        landmark = np.load(self.landmarks_list[item])[0]
        label = self.label_list[item]

        # get bounding
        bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])
        w = bbox_lm[2] - bbox_lm[0]
        h = bbox_lm[3] - bbox_lm[1]
        x0 = int(max(bbox_lm[0] - w * 0.1, 0))
        y0 = int(max(bbox_lm[1] - h * 0.1, 0))
        x1 = int(min(bbox_lm[2] + w * 0.1, image.shape[1]))
        y1 = int(min(bbox_lm[3] + h * 0.1, image.shape[0]))
        bbox = np.array([[x0, y0], [x1, y1]])

        # re-order landmark
        landmark = reorder_landmark(landmark)

        # horizontal flip
        if self.phase == 'train':
            if np.random.rand() < 0.5:
                image, _, landmark, bbox = hflip(image, None, landmark, bbox)

        # crop landmarks
        image, landmark, bbox, __ = crop_face(image, landmark, bbox, margin=True, crop_by_bbox=False)

        # Contrastive Transform
        image = self.contrast_transform(Image.fromarray(image))
        types = self.type_list[item]

        return image, label, types

def one_hot_encoding(label:int, num_classes=2):
    vec = torch.zeros(size=(num_classes,), dtype=torch.float32)
    vec[label] = 1.
    return vec

def reorder_landmark(landmark):
    landmark_add = np.zeros((13, 2))
    for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
        landmark_add[idx] = landmark[idx_l]
    landmark[68:] = landmark_add
    return landmark

def hflip(img, mask=None, landmark=None, bbox=None):
    H, W = img.shape[:2]
    landmark = landmark.copy()
    bbox = bbox.copy()

    if landmark is not None:
        landmark_new = np.zeros_like(landmark)

        landmark_new[:17] = landmark[:17][::-1]
        landmark_new[17:27] = landmark[17:27][::-1]

        landmark_new[27:31] = landmark[27:31]
        landmark_new[31:36] = landmark[31:36][::-1]

        landmark_new[36:40] = landmark[42:46][::-1]
        landmark_new[40:42] = landmark[46:48][::-1]

        landmark_new[42:46] = landmark[36:40][::-1]
        landmark_new[46:48] = landmark[40:42][::-1]

        landmark_new[48:55] = landmark[48:55][::-1]
        landmark_new[55:60] = landmark[55:60][::-1]

        landmark_new[60:65] = landmark[60:65][::-1]
        landmark_new[65:68] = landmark[65:68][::-1]
        if len(landmark) == 68:
            pass
        elif len(landmark) == 81:
            landmark_new[68:81] = landmark[68:81][::-1]
        else:
            raise NotImplementedError
        landmark_new[:, 0] = W - landmark_new[:, 0]

    else:
        landmark_new = None

    if bbox is not None:
        bbox_new = np.zeros_like(bbox)
        bbox_new[0, 0] = bbox[1, 0]
        bbox_new[1, 0] = bbox[0, 0]
        bbox_new[:, 0] = W - bbox_new[:, 0]
        bbox_new[:, 1] = bbox[:, 1].copy()
        if len(bbox) > 2:
            bbox_new[2, 0] = W - bbox[3, 0]
            bbox_new[2, 1] = bbox[3, 1]
            bbox_new[3, 0] = W - bbox[2, 0]
            bbox_new[3, 1] = bbox[2, 1]
            bbox_new[4, 0] = W - bbox[4, 0]
            bbox_new[4, 1] = bbox[4, 1]
            bbox_new[5, 0] = W - bbox[6, 0]
            bbox_new[5, 1] = bbox[6, 1]
            bbox_new[6, 0] = W - bbox[5, 0]
            bbox_new[6, 1] = bbox[5, 1]
    else:
        bbox_new = None

    if mask is not None:
        mask = mask[:, ::-1]
    else:
        mask = None
    img = img[:, ::-1].copy()
    return img, mask, landmark_new, bbox_new

class LinearDataset(Dataset):
    def __init__(self, dataset_configs, phase="train", transform=None):
        self.configs = dataset_configs
        self.phase = phase
        self.transform = transform
        self.filelist = pd.DataFrame(columns=["Image Path", "Landmarks Path", "Video", "Label", "Type"])

        # read file list
        for csv_fp in data_csv:
            self.filelist = pd.concat([self.filelist, pd.read_csv(csv_fp)], ignore_index=True)
        self.filelist = self.filelist.sample(frac=1).reset_index()
        self.image_list = np.array(self.filelist["Image Path"])
        self.landmarks_list = np.array(self.filelist["Landmarks Path"])
        self.video_list = np.array(self.filelist["Video"])
        self.label_list = np.array(self.filelist["Label"])
        self.type_list = np.array(self.filelist["Type"])

        print(f"[Dataset] Init successfully!!")

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item):
        # read image, landmarks and label
        image = cv2.imread(self.image_list[item], cv2.IMREAD_UNCHANGED)
        im = image
        landmark = np.load(self.landmarks_list[item])[0]
        label = self.label_list[item]

        # get bounding
        bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])
        w = bbox_lm[2] - bbox_lm[0]
        h = bbox_lm[3] - bbox_lm[1]
        x0 = int(max(bbox_lm[0] - w * 0.1, 0))
        y0 = int(max(bbox_lm[1] - h * 0.1, 0))
        x1 = int(min(bbox_lm[2] + w * 0.1, image.shape[1]))
        y1 = int(min(bbox_lm[3] + h * 0.1, image.shape[0]))
        bbox = np.array([[x0, y0], [x1, y1]])

        # re-order landmark
        landmark = reorder_landmark(landmark)

        # horizontal flip
        if self.phase == 'train':
            if np.random.rand() < 0.5:
                image, _, landmark, bbox = hflip(image, None, landmark, bbox)

        # crop landmarks
        image, landmark, bbox, __ = crop_face(image, landmark, bbox, margin=True, crop_by_bbox=False)

        # Contrastive Transform
        if self.transform is not None:
            image = self.transform(Image.fromarray(image))

        # label = one_hot_encoding(label, 2)
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

        # label = one_hot_encoding(label, 2)
        return image, label


class CelebSupConDataset(Dataset):
    def __init__(self, dataset_configs, phase="test", transform=None):
        self.configs = dataset_configs
        self.phase = phase
        self.contrast_transform = TwoCropTransform(transform)

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

        if self.contrast_transform is not None:
            image = self.contrast_transform(Image.fromarray(image))

        # label = one_hot_encoding(label, 2)
        return image, label



class PreprocessedFaceForencisDataset(Dataset):
    def __init__(self, configs, phase="train", transform=None, using_contrast=False):
        super(Dataset, self).__init__()
        self.configs = configs
        self.phase = phase
        self.using_contrast = using_contrast

        if transform is None:
            self.transform = default_transform
        else:
            self.transform = transform

        if using_contrast:
            self.contrast_transform = TwoCropTransform(transform)
        else:
            self.contrast_transform = transform

        if self.phase == "train":
            self.root_folder = self.configs.dataset.train_root
        elif self.phase in ["test", "valid"]:
            self.root_folder = self.configs.dataset.test_root
        else:
            raise "Phase is not valid"

        images_folder = os.path.join(self.root_folder, "images")
        landmarks_folder = os.path.join(self.root_folder, "landmarks")

        list_folder = {  # folder and its label
            "original": 1,
            "DeepFakeDetection": 0,
            "FaceSwap": 0,
            "Face2Face": 0,
            "FaceShifter": 0,
            "NeuralTextures": 0,
            "Deepfakes": 0,
        }

        self.images_file_list = []
        self.landmarks_file_list = []
        self.labels_list = []

        for k, v in list_folder.items():
            images_folder_path = os.path.join(images_folder, k)
            landmarks_folder_path = os.path.join(landmarks_folder, k)

            video_list = os.listdir(images_folder_path)
            images_video_list = [os.path.join(images_folder_path, video) for video in video_list]
            landmarks_video_list = [os.path.join(landmarks_folder_path, video) for video in video_list]

            for images_video, landmarks_video in zip(images_video_list, landmarks_video_list):

                image_samples_list = os.listdir(images_video)
                image_samples_list.sort()
                landmark_samples_list = os.listdir(landmarks_video)
                landmark_samples_list.sort()

                n = len(image_samples_list)
                assert n == len(landmark_samples_list)

                self.images_file_list = self.images_file_list + [os.path.join(images_video, fn) for fn in
                                                                 image_samples_list]
                self.landmarks_file_list = self.landmarks_file_list + [os.path.join(landmarks_video, fn) for fn in
                                                                       landmark_samples_list]
                self.labels_list += [v for _ in range(n)]

    def __len__(self):
        return len(self.images_file_list)

    def __getitem__(self, item):
        assert self.landmarks_file_list[item].split("/")[-1].split(".")[0] == \
               self.images_file_list[item].split("/")[-1].split(".")[0]

        # read image, landmarks and label
        image = cv2.imread(self.images_file_list[item], cv2.IMREAD_UNCHANGED)
        landmark = np.load(self.landmarks_file_list[item])
        label = self.labels_list[item]

        # get bounding
        bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])
        w = bbox_lm[2] - bbox_lm[0]
        h = bbox_lm[3] - bbox_lm[1]
        x0 = int(max(bbox_lm[0] - w * 0.1, 0))
        y0 = int(max(bbox_lm[1] - h * 0.1, 0))
        x1 = int(min(bbox_lm[2] + w * 0.1, image.shape[1]))
        y1 = int(min(bbox_lm[3] + h * 0.1, image.shape[0]))
        bbox = np.array([[x0, y0], [x1, y1]])

        # re-order landmark
        landmark = reorder_landmark(landmark)

        # horizontal flip
        if self.phase == 'train':
            if np.random.rand() < 0.5:
                image, _, landmark, bbox = hflip(image, None, landmark, bbox)

        # Contrastive Transform
        image = self.contrast_transform(Image.fromarray(image))

        return image, label


if __name__ == "__main__":
    from yacs.config import CfgNode
    class A:
        dataset = CfgNode()
        dataset.train_root = "/mnt/data/duongdhk/datasets/processed_deepfake_detection_dataset/processed_FFPP"

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = PreprocessedFaceForencisDataset(A, "train", train_transform, using_contrast=True)
    for data in dataset:
        print(data)