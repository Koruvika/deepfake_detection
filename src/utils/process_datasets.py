import glob
import os.path
import random

import cv2
import numpy as np
import pandas as pd


def get_list(src, des, path, name, label, n_frame=5):
    result = []
    for video_fn in sorted(os.listdir(os.path.join(src, path, "frames"))):
        image_list, landmarks_list = sorted(os.listdir(os.path.join(src, path, "frames", video_fn))), sorted(os.listdir(os.path.join(src, path, "landmarks", video_fn)))

        # random shuffle split
        indices = np.linspace(0, len(image_list) - 1, len(image_list)).astype(np.int32)
        np.random.shuffle(indices)
        n_frame = min(n_frame, len(image_list))
        indices = indices[:n_frame]
        image_list = np.array(image_list)[indices]
        landmarks_list = np.array(landmarks_list)[indices]

        # loop and save
        for image_fn, landmarks_fn in zip(image_list, landmarks_list):
            image_fp = os.path.join(src, path, "frames", video_fn, image_fn)
            landmarks_fp = os.path.join(src, path, "landmarks", video_fn, landmarks_fn)
            result.append([image_fp, landmarks_fp, video_fn, label, name])

    df = pd.DataFrame(result, columns=["Image Path", "Landmarks Path", "Video", "Label", "Type"])
    df.to_csv(os.path.join(des, f"{name}.csv"))
    print(len(df))

def get_celeb_lips(src, des, path, name, label):
    result = []
    for video_fn in sorted(os.listdir(os.path.join(src, path, "frames"))):
        image_list, landmarks_list = sorted(os.listdir(os.path.join(src, path, "frames", video_fn))), sorted(os.listdir(os.path.join(src, path, "landmarks", video_fn)))

        # loop and save
        for image_fn, landmarks_fn in zip(image_list, landmarks_list):
            image_fp = os.path.join(src, path, "frames", video_fn, image_fn)
            landmarks_fp = os.path.join(src, path, "landmarks", video_fn, landmarks_fn)
            result.append([image_fp, landmarks_fp, video_fn, label, name])

    df = pd.DataFrame(result, columns=["Image Path", "Landmarks Path", "Video", "Label", "Type"])
    df.to_csv(os.path.join(des, f"{name}.csv"))
    print(len(df))



def process_ffpp():
    src_root = "/mnt/data/duongdhk/datasets/FFPP"
    des_root = "/mnt/data/duongdhk/datasets/processed_deepfake_detection_dataset/FFPP"

    get_list(src_root, des_root, "original_sequences/youtube/c23", "original", 1, 30)
    get_list(src_root, des_root, "manipulated_sequences/DeepFakeDetection/c23", "DeepFakeDetection", 0, 5)
    get_list(src_root, des_root, "manipulated_sequences/Deepfakes/c23", "Deepfakes", 0, 5)
    get_list(src_root, des_root, "manipulated_sequences/Face2Face/c23", "Face2Face", 0, 5)
    get_list(src_root, des_root, "manipulated_sequences/FaceShifter/c23", "FaceShifter", 0, 5)
    get_list(src_root, des_root, "manipulated_sequences/FaceSwap/c23", "FaceSwap", 0, 5)
    get_list(src_root, des_root, "manipulated_sequences/NeuralTextures/c23", "NeuralTextures", 0, 5)


def load_ffpp(list_folder, data_rate):
    """
    Load images, labels [0: fake, 1: real], type [original, deepfake, faceswap, ...] and landmarks
    """
    results = dict()
    for k, n in data_rate.items():
        csv_path = os.path.join(list_folder, f"{k}.csv")
        df = pd.read_csv(csv_path)
        result = df.sample(n=n, weights="Video")
        results[k] = result
    return results



    # get_celeb_lips()

if __name__ == "__main__":
    load_ffpp()