import os

import cv2
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

sys.path.insert(0, "")
from src.utils import crop_face


def draw_landmarks(img, landmarks, c=(255, 0, 0)):
    img = np.ascontiguousarray(img)
    for landmark in landmarks:
        x = int(landmark[0])
        y = int(landmark[1])
        img = cv2.circle(img=img, center=(x, y), radius=1, color=c, thickness=1)
    return img


def reorder_landmark(landmark):
    landmark_add = np.zeros((13, 2))
    for idx, idx_l in enumerate([77, 75, 76, 68, 69, 70, 71, 80, 72, 73, 79, 74, 78]):
        landmark_add[idx] = landmark[idx_l]
    landmark[68:] = landmark_add
    return landmark

def main():
    save_root = "/mnt/data/duongdhk/datasets/processed_deepfake_detection_dataset/processed_FFPP"
    data_csv = {
        "original": "original.csv",
        "DeepFakeDetection": "DeepFakeDetection.csv",
        "Deepfakes": "Deepfakes.csv",
        "Face2Face": "Face2Face.csv",
        "FaceShifter": "FaceShifter.csv",
        "FaceSwap": "FaceSwap.csv",
        "NeuralTextures": "NeuralTextures.csv",
    }
    data_csv = [os.path.join("/mnt/data/duongdhk/datasets/processed_deepfake_detection_dataset/FFPP", fp) for fp in
                data_csv.values()]
    filelist = pd.DataFrame(columns=["Image Path", "Landmarks Path", "Video", "Label", "Type"])
    for csv_fp in data_csv:
        filelist = pd.concat([filelist, pd.read_csv(csv_fp)], ignore_index=True)
    filelist = filelist.sample(frac=1).reset_index()
    image_list = np.array(filelist["Image Path"])
    video_list = np.array(filelist["Video"])
    landmarks_list = np.array(filelist["Landmarks Path"])
    label_list = np.array(filelist["Label"])
    type_list = np.array(filelist["Type"])


    for item in tqdm(range(len(filelist))):
        image = cv2.imread(image_list[item], cv2.IMREAD_UNCHANGED)
        landmark = np.load(landmarks_list[item])[0]
        video = video_list[item]
        ty = type_list[item]

        bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])
        w = bbox_lm[2] - bbox_lm[0]
        h = bbox_lm[3] - bbox_lm[1]
        x0 = int(max(bbox_lm[0] - w * 0.1, 0))
        y0 = int(max(bbox_lm[1] - h * 0.1, 0))
        x1 = int(min(bbox_lm[2] + w * 0.1, image.shape[1]))
        y1 = int(min(bbox_lm[3] + h * 0.1, image.shape[0]))
        bbox = np.array([[x0, y0], [x1, y1]])

        # landmark = reorder_landmark(landmark)
        image, landmark, bbox, __ = crop_face(image, landmark, bbox, margin=True, crop_by_bbox=False)
        os.makedirs(os.path.join(save_root, "images", ty, str(video)), exist_ok=True)
        os.makedirs(os.path.join(save_root, "landmarks", ty, str(video)), exist_ok=True)
        save_path = os.path.join(save_root, "images", ty, str(video), image_list[item].split("/")[-1])

        assert cv2.imwrite(save_path, image)
        np.save(os.path.join(save_root, "landmarks", ty, str(video), image_list[item].split("/")[-1][:-4] + ".npy"), landmark)


def process_celeb():
    src_root = "/mnt/data/duongdhk/datasets/Celeb-DF-v2"
    save_root = "/mnt/data/duongdhk/datasets/processed_deepfake_detection_dataset/Celeb-DF-v2"

    list_path = "/mnt/data/duongdhk/datasets/Celeb-DF-v2/List_of_testing_videos.txt"

    with open(list_path, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        label, mp4_file = line.split(" ")
        video = mp4_file.split(".")[0]
        ty, video = video.split("/")

        image_folder_path = os.path.join(src_root, ty, "frames", video)
        landmarks_folder_path = os.path.join(src_root, ty, "landmarks", video)

        images_list = sorted(os.listdir(image_folder_path))
        landmarks_list = sorted(os.listdir(landmarks_folder_path))

        # images_list = [os.path.join(image_folder_path, im) for im in images_list]
        # landmarks_list = [os.path.join(image_folder_path, ld) for ld in landmarks_list]

        for im_fn, ld_fn in zip(images_list, landmarks_list):
            im_fp = os.path.join(image_folder_path, im_fn)
            ld_fp = os.path.join(landmarks_folder_path, ld_fn)

            image = cv2.imread(im_fp, cv2.IMREAD_UNCHANGED)
            landmark = np.load(ld_fp)[0]

            bbox_lm = np.array([landmark[:, 0].min(), landmark[:, 1].min(), landmark[:, 0].max(), landmark[:, 1].max()])
            w = bbox_lm[2] - bbox_lm[0]
            h = bbox_lm[3] - bbox_lm[1]
            x0 = int(max(bbox_lm[0] - w * 0.1, 0))
            y0 = int(max(bbox_lm[1] - h * 0.1, 0))
            x1 = int(min(bbox_lm[2] + w * 0.1, image.shape[1]))
            y1 = int(min(bbox_lm[3] + h * 0.1, image.shape[0]))
            bbox = np.array([[x0, y0], [x1, y1]])
            image, landmark, bbox, __ = crop_face(image, landmark, bbox, margin=True, crop_by_bbox=False)

            os.makedirs(os.path.join(save_root, "images", ty, str(video)), exist_ok=True)
            os.makedirs(os.path.join(save_root, "landmarks", ty, str(video)), exist_ok=True)
            save_path = os.path.join(save_root, "images", ty, str(video), im_fn)
            assert cv2.imwrite(save_path, image)
            np.save(os.path.join(save_root, "landmarks", ty, str(video), ld_fn), landmark)


if __name__ == "__main__":
    process_celeb()