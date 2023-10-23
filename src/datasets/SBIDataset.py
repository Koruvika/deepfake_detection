import logging
import random

import cv2
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
import os
import json
import numpy as np
import albumentations as alb
# import sys
import torch
#
# sys.path.insert(0, "")
from src.utils import RandomDownScale, random_get_hull, IoUfrom2bboxes, crop_face, dynamic_blend


def draw_landmarks(img_, landmarks):
    img = img_.copy()
    for landmark in landmarks:
        x = int(landmark[0])
        y = int(landmark[1])
        img = cv2.circle(img, center=(x, y), radius=1, color=(255, 0, 0), thickness=3)

    return img


def init_face_forencis(phase, dataset_path, level='frame', n_frames=8):
    image_list = []
    label_list = []

    list_path = os.path.join(dataset_path, f"{phase}.json")
    frames_path = os.path.join(dataset_path, "original_sequences/youtube/c23/frames/")

    folder_list = sorted(glob(frames_path+'*'))
    filelist = []
    list_dict = json.load(open(list_path,'r'))
    for i in list_dict:
        filelist+=i
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]

    if level =='video':
        label_list=[0]*len(folder_list)
        return folder_list,label_list
    for i in range(len(folder_list)):
        # images_temp=sorted([glob(folder_list[i]+'/*.png')[0]])
        images_temp=sorted(glob(folder_list[i]+'/*.png'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
        image_list+=images_temp
        label_list+=[0]*len(images_temp)

    return image_list,label_list


class SBIFaceForencisDataset(Dataset):
    def __init__(self, dataset_path: str, phase: str = "train", image_size: int = 224, n_frames:int = 8):
        super(Dataset, self).__init__()
        assert phase in ["train", "val", "test"]
        image_list, label_list = init_face_forencis(phase=phase, dataset_path=dataset_path, level='frame',
                                                    n_frames=n_frames)

        path_lm = '/landmarks/'
        label_list = [label_list[i] for i in range(len(image_list)) if os.path.isfile(
            image_list[i].replace('/frames/', path_lm).replace('.png', '.npy')) or os.path.isfile(
            image_list[i].replace('/frames/', '/retina/').replace('.png', '.npy'))]
        image_list = [image_list[i] for i in range(len(image_list)) if os.path.isfile(
            image_list[i].replace('/frames/', path_lm).replace('.png', '.npy')) or os.path.isfile(
            image_list[i].replace('/frames/', '/retina/').replace('.png', '.npy'))]
        self.path_lm = path_lm
        print(f'SBI({phase}): {len(image_list)}')

        self.image_list = image_list
        self.image_size = (image_size, image_size)
        self.phase = phase
        self.n_frames = n_frames

        self.transforms = self.get_transforms()
        self.source_transforms = self.get_source_transforms()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        flag = True
        while flag:
            try:
                filename = self.image_list[idx]
                img = np.array(Image.open(filename))
                landmark = np.load(filename.replace('.png', '.npy').replace('/frames/', self.path_lm))[0]
                bbox_lm = np.array(
                    [landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])
                # bboxes = np.load(filename.replace('.png', '.npy').replace('/frames/', '/retina/'))[:2]
                w = bbox_lm[2] - bbox_lm[0]
                h = bbox_lm[3] - bbox_lm[1]
                x0 = int(max(bbox_lm[0] - w * 0.1, 0))
                y0 = int(max(bbox_lm[1] - h * 0.1, 0))
                x1 = int(min(bbox_lm[2] + w * 0.1, img.shape[1]))
                y1 = int(min(bbox_lm[3] + h * 0.1, img.shape[0]))

                bbox = np.array([[x0, y0], [x1, y1]])
                # bboxes = np.array([[x0, y0, x1, y1]]) #  [x0, y0, x1, y1]
                # iou_max = -1
                # for i in range(len(bboxes)):
                #     iou = IoUfrom2bboxes(bbox_lm, bboxes[i].flatten())
                #     if iou_max < iou:
                #         bbox = bboxes[i]
                #         iou_max = iou

                landmark = self.reorder_landmark(landmark)
                if self.phase == 'train':
                    if np.random.rand() < 0.5:
                        img, _, landmark, bbox = self.hflip(img, None, landmark, bbox)

                img, landmark, bbox, __ = crop_face(img, landmark, bbox, margin=True, crop_by_bbox=False)

                img_r, img_f, mask_f = self.self_blending(img.copy(), landmark.copy())

                if self.phase == 'train':
                    transformed = self.transforms(image=img_f.astype('uint8'), image1=img_r.astype('uint8'))
                    img_f = transformed['image']
                    img_r = transformed['image1']

                img_f, _, __, ___, y0_new, y1_new, x0_new, x1_new = crop_face(img_f, landmark, bbox.reshape(2, 2), margin=False,
                                                                              crop_by_bbox=True, abs_coord=True,
                                                                              phase=self.phase)

                img_r = img_r[y0_new:y1_new, x0_new:x1_new]

                img_f = cv2.resize(img_f, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32') / 255
                img_r = cv2.resize(img_r, self.image_size, interpolation=cv2.INTER_LINEAR).astype('float32') / 255

                img_f = img_f.transpose((2, 0, 1))
                img_r = img_r.transpose((2, 0, 1))
                flag = False
            except Exception as e:
                print(e)
                idx = random.randint(0, len(self))

        return img_f, img_r

    def get_source_transforms(self):
        return alb.Compose([
            alb.Compose([
                alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
                alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                       val_shift_limit=(-0.3, 0.3), p=1),
                alb.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=1),
            ], p=1),

            alb.OneOf([
                RandomDownScale(p=1),
                alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1),
            ], p=1),

        ], p=1.)

    def get_transforms(self):
        return alb.Compose([
            alb.RGBShift((-20, 20), (-20, 20), (-20, 20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3, 0.3), sat_shift_limit=(-0.3, 0.3),
                                   val_shift_limit=(-0.3, 0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
            alb.ImageCompression(quality_lower=40, quality_upper=100, p=0.5),
        ], additional_targets={f'image1': 'image'}, p=1.)


    def randaffine(self, img, mask):
        f = alb.Affine(
            translate_percent={'x': (-0.03, 0.03), 'y': (-0.015, 0.015)},
            scale=[0.95, 1 / 0.95],
            fit_output=False,
            p=1)

        g = alb.ElasticTransform(
            alpha=50,
            sigma=7,
            alpha_affine=0,
            p=1,
        )

        transformed = f(image=img, mask=mask)
        img = transformed['image']

        mask = transformed['mask']
        transformed = g(image=img, mask=mask)
        mask = transformed['mask']
        return img, mask

    def self_blending(self, img, landmark, exist_bi=True):
        H, W = len(img), len(img[0])
        if np.random.rand() < 0.25:
            landmark = landmark[:68]
        if exist_bi:
            logging.disable(logging.FATAL)
            mask = random_get_hull(landmark, img)[:, :, 0]
            logging.disable(logging.NOTSET)
        else:
            mask = np.zeros_like(img[:, :, 0])
            cv2.fillConvexPoly(mask, cv2.convexHull(landmark), 1.)

        source = img.copy()
        if np.random.rand() < 0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source, mask)

        img_blended, mask = dynamic_blend(source, img, mask)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img, img_blended, mask

    def reorder_landmark(self, landmark):
        landmark_add = np.zeros((13, 2))
        for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
            landmark_add[idx] = landmark[idx_l]
        landmark[68:] = landmark_add
        return landmark

    def hflip(self, img, mask=None, landmark=None, bbox=None):
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

    def collate_fn(self, batch):
        img_f, img_r = zip(*batch)
        data = {}
        # data['img'] = torch.cat((torch.tensor(img_r).float(), torch.tensor(img_f).float()), 0)
        data['img'] = torch.from_numpy(np.concatenate((img_r, img_f), axis=0, dtype=np.float32))
        data['label'] = torch.from_numpy(np.array([0] * len(img_r) + [1] * len(img_f)))
        return data

    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)



if __name__ == "__main__":
    dataset = SBIFaceForencisDataset()
    