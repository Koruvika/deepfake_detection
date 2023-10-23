import inspect
import json
import logging
import os
import random
import sys

import albumentations as alb
import cv2
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa
from skimage import io
from skimage import transform as sktransform

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self, img, **params):
        return self.randomdownscale(img)

    def randomdownscale(self, img):
        keep_ratio = True
        keep_input_shape = True
        H, W, C = img.shape
        ratio_list = [2, 4]
        r = ratio_list[np.random.randint(len(ratio_list))]
        img_ds = cv2.resize(img, (int(W / r), int(H / r)), interpolation=cv2.INTER_NEAREST)
        if keep_input_shape:
            img_ds = cv2.resize(img_ds, (W, H), interpolation=cv2.INTER_LINEAR)

        return img_ds


def get_available_masks():
    """ Return a list of the available masks for cli """
    masks = sorted([name for name, obj in inspect.getmembers(sys.modules[__name__])
                    if inspect.isclass(obj) and name != "Mask"])
    masks.append("none")
    logger.debug(masks)
    return masks


def get_default_mask():
    """ Set the default mask for cli """
    masks = get_available_masks()
    default = "dfl_full"
    default = default if default in masks else masks[0]
    logger.debug(default)
    return default


class Mask:
    """ Parent class for masks
        the output mask will be <mask_type>.mask
        channels: 1, 3 or 4:
                    1 - Returns a single channel mask
                    3 - Returns a 3 channel mask
                    4 - Returns the original image with the mask in the alpha channel """

    def __init__(self, landmarks, face, channels=4):
        logger.info("Initializing %s: (face_shape: %s, channels: %s, landmarks: %s)",
                    self.__class__.__name__, face.shape, channels, landmarks)
        self.landmarks = landmarks
        self.face = face
        self.channels = channels

        mask = self.build_mask()
        self.mask = self.merge_mask(mask)
        logger.info("Initialized %s", self.__class__.__name__)

    def build_mask(self):
        """ Override to build the mask """
        raise NotImplementedError

    def merge_mask(self, mask):
        """ Return the mask in requested shape """
        logger.info("mask_shape: %s", mask.shape)
        assert self.channels in (1, 3, 4), "Channels should be 1, 3 or 4"
        assert mask.shape[2] == 1 and mask.ndim == 3, "Input mask be 3 dimensions with 1 channel"

        if self.channels == 3:
            retval = np.tile(mask, 3)
        elif self.channels == 4:
            retval = np.concatenate((self.face, mask), -1)
        else:
            retval = mask

        logger.info("Final mask shape: %s", retval.shape)
        return retval


class dfl_full(Mask):  # pylint: disable=invalid-name
    """ DFL facial mask """

    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1,), dtype=np.float32)

        nose_ridge = (self.landmarks[27:31], self.landmarks[33:34])
        jaw = (self.landmarks[0:17],
               self.landmarks[48:68],
               self.landmarks[0:1],
               self.landmarks[8:9],
               self.landmarks[16:17])
        eyes = (self.landmarks[17:27],
                self.landmarks[0:1],
                self.landmarks[27:28],
                self.landmarks[16:17],
                self.landmarks[33:34])
        parts = [jaw, nose_ridge, eyes]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class components(Mask):  # pylint: disable=invalid-name
    """ Component model mask """

    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1,), dtype=np.float32)

        r_jaw = (self.landmarks[0:9], self.landmarks[17:18])
        l_jaw = (self.landmarks[8:17], self.landmarks[26:27])
        r_cheek = (self.landmarks[17:20], self.landmarks[8:9])
        l_cheek = (self.landmarks[24:27], self.landmarks[8:9])
        nose_ridge = (self.landmarks[19:25], self.landmarks[8:9],)
        r_eye = (self.landmarks[17:22],
                 self.landmarks[27:28],
                 self.landmarks[31:36],
                 self.landmarks[8:9])
        l_eye = (self.landmarks[22:27],
                 self.landmarks[27:28],
                 self.landmarks[31:36],
                 self.landmarks[8:9])
        nose = (self.landmarks[27:31], self.landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class extended(Mask):  # pylint: disable=invalid-name
    """ Extended mask
        Based on components mask. Attempts to extend the eyebrow points up the forehead
    """

    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1,), dtype=np.float32)

        landmarks = self.landmarks.copy()
        # mid points between the side of face and eye point
        ml_pnt = (landmarks[36] + landmarks[0]) // 2
        mr_pnt = (landmarks[16] + landmarks[45]) // 2

        # mid points between the mid points and eye
        ql_pnt = (landmarks[36] + ml_pnt) // 2
        qr_pnt = (landmarks[45] + mr_pnt) // 2

        # Top of the eye arrays
        bot_l = np.array((ql_pnt, landmarks[36], landmarks[37], landmarks[38], landmarks[39]))
        bot_r = np.array((landmarks[42], landmarks[43], landmarks[44], landmarks[45], qr_pnt))

        # Eyebrow arrays
        top_l = landmarks[17:22]
        top_r = landmarks[22:27]

        # Adjust eyebrow arrays
        landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
        landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

        r_jaw = (landmarks[0:9], landmarks[17:18])
        l_jaw = (landmarks[8:17], landmarks[26:27])
        r_cheek = (landmarks[17:20], landmarks[8:9])
        l_cheek = (landmarks[24:27], landmarks[8:9])
        nose_ridge = (landmarks[19:25], landmarks[8:9],)
        r_eye = (landmarks[17:22], landmarks[27:28], landmarks[31:36], landmarks[8:9])
        l_eye = (landmarks[22:27], landmarks[27:28], landmarks[31:36], landmarks[8:9])
        nose = (landmarks[27:31], landmarks[31:36])
        parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

        for item in parts:
            merged = np.concatenate(item)
            cv2.fillConvexPoly(mask, cv2.convexHull(merged), 255.)  # pylint: disable=no-member
        return mask


class facehull(Mask):  # pylint: disable=invalid-name
    """ Basic face hull mask """

    def build_mask(self):
        mask = np.zeros(self.face.shape[0:2] + (1,), dtype=np.float32)
        hull = cv2.convexHull(  # pylint: disable=no-member
            np.array(self.landmarks).reshape((-1, 2)))
        cv2.fillConvexPoly(mask, hull, 255.0, lineType=cv2.LINE_AA)  # pylint: disable=no-member
        return mask


def name_resolve(path):
    name = os.path.splitext(os.path.basename(path))[0]
    vid_id, frame_id = name.split('_')[0:2]
    return vid_id, frame_id


def total_euclidean_distance(a, b):
    assert len(a.shape) == 2
    return np.sum(np.linalg.norm(a - b, axis=1))


def random_get_hull(landmark, img1):
    hull_type = random.choice([0, 1, 2, 3])
    if hull_type == 0:
        mask = dfl_full(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
        return mask / 255
    elif hull_type == 1:
        mask = extended(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
        return mask / 255
    elif hull_type == 2:
        mask = components(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
        return mask / 255
    elif hull_type == 3:
        mask = facehull(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
        return mask / 255


def random_erode_dilate(mask, ksize=None):
    if random.random() > 0.5:
        if ksize is None:
            ksize = random.randint(1, 21)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8) * 255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.erode(mask, kernel, 1) / 255
    else:
        if ksize is None:
            ksize = random.randint(1, 5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8) * 255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.dilate(mask, kernel, 1) / 255
    return mask


# borrow from https://github.com/MarekKowalski/FaceSwap
def blendImages(src, dst, mask, featherAmount=0.2):
    maskIndices = np.where(mask != 0)

    src_mask = np.ones_like(mask)
    dst_mask = np.zeros_like(mask)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (
            1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    composedMask = np.copy(dst_mask)
    composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[maskIndices[0], maskIndices[1]] + (
            1 - weights[:, np.newaxis]) * dst_mask[maskIndices[0], maskIndices[1]]

    return composedImg, composedMask


# borrow from https://github.com/MarekKowalski/FaceSwap
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)

    maskIndices = np.where(mask != 0)

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst


class BIOnlineGeneration:
    def __init__(self):
        with open('precomuted_landmarks.json', 'r') as f:
            self.landmarks_record = json.load(f)
            for k, v in self.landmarks_record.items():
                self.landmarks_record[k] = np.array(v)
        # extract all frame from all video in the name of {videoid}_{frameid}
        self.data_list = [
                             '000_0000.png',
                             '001_0000.png'
                         ] * 10000

        # predefine mask distortion
        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])

    def gen_one_datapoint(self):
        background_face_path = random.choice(self.data_list)
        data_type = 'real' if random.randint(0, 1) else 'fake'
        if data_type == 'fake':
            face_img, mask = self.get_blended_face(background_face_path)
            mask = (1 - mask) * mask * 4
        else:
            face_img = io.imread(background_face_path)
            mask = np.zeros((317, 317, 1))

        # randomly downsample after BI pipeline
        if random.randint(0, 1):
            aug_size = random.randint(64, 317)
            face_img = Image.fromarray(face_img)
            if random.randint(0, 1):
                face_img = face_img.resize((aug_size, aug_size), Image.BILINEAR)
            else:
                face_img = face_img.resize((aug_size, aug_size), Image.NEAREST)
            face_img = face_img.resize((317, 317), Image.BILINEAR)
            face_img = np.array(face_img)

        # random jpeg compression after BI pipeline
        if random.randint(0, 1):
            quality = random.randint(60, 100)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            face_img_encode = cv2.imencode('.jpg', face_img, encode_param)[1]
            face_img = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)

        face_img = face_img[60:317, 30:287, :]
        mask = mask[60:317, 30:287, :]

        # random flip
        if random.randint(0, 1):
            face_img = np.flip(face_img, 1)
            mask = np.flip(mask, 1)

        return face_img, mask, data_type

    def get_blended_face(self, background_face_path):
        background_face = io.imread(background_face_path)
        background_landmark = self.landmarks_record[background_face_path]

        foreground_face_path = self.search_similar_face(background_landmark, background_face_path)
        foreground_face = io.imread(foreground_face_path)

        # down sample before blending
        aug_size = random.randint(128, 317)
        background_landmark = background_landmark * (aug_size / 317)
        foreground_face = sktransform.resize(foreground_face, (aug_size, aug_size), preserve_range=True).astype(
            np.uint8)
        background_face = sktransform.resize(background_face, (aug_size, aug_size), preserve_range=True).astype(
            np.uint8)

        # get random type of initial blending mask
        mask = random_get_hull(background_landmark, background_face)

        #  random deform mask
        mask = self.distortion.augment_image(mask)
        mask = random_erode_dilate(mask)

        # filte empty mask after deformation
        if np.sum(mask) == 0:
            raise NotImplementedError

        # apply color transfer
        foreground_face = colorTransfer(background_face, foreground_face, mask * 255)

        # blend two face
        blended_face, mask = blendImages(foreground_face, background_face, mask * 255)
        blended_face = blended_face.astype(np.uint8)

        # resize back to default resolution
        blended_face = sktransform.resize(blended_face, (317, 317), preserve_range=True).astype(np.uint8)
        mask = sktransform.resize(mask, (317, 317), preserve_range=True)
        mask = mask[:, :, 0:1]
        return blended_face, mask

    def search_similar_face(self, this_landmark, background_face_path):
        vid_id, frame_id = name_resolve(background_face_path)
        min_dist = 99999999

        # random sample 5000 frame from all frams:
        all_candidate_path = random.sample(self.data_list, k=5000)

        # filter all frame that comes from the same video as background face
        all_candidate_path = filter(lambda k: name_resolve(k)[0] != vid_id, all_candidate_path)
        all_candidate_path = list(all_candidate_path)

        # loop throungh all candidates frame to get best match
        min_path = None
        for candidate_path in all_candidate_path:
            candidate_landmark = self.landmarks_record[candidate_path].astype(np.float32)
            candidate_distance = total_euclidean_distance(candidate_landmark, this_landmark)
            if candidate_distance < min_dist:
                min_dist = candidate_distance
                min_path = candidate_path

        return min_path


def alpha_blend(source, target, mask):
    mask_blured = get_blend_mask(mask)
    img_blended = (mask_blured * source + (1 - mask_blured) * target)
    return img_blended, mask_blured


def dynamic_blend(source, target, mask):
    mask_blured = get_blend_mask(mask)
    blend_list = [0.25, 0.5, 0.75, 1, 1, 1]
    blend_ratio = blend_list[np.random.randint(len(blend_list))]
    mask_blured *= blend_ratio
    img_blended = (mask_blured * source + (1 - mask_blured) * target)
    return img_blended, mask_blured


def get_blend_mask(mask):
    H, W = mask.shape
    size_h = np.random.randint(192, 257)
    size_w = np.random.randint(192, 257)
    mask = cv2.resize(mask, (size_w, size_h))
    kernel_1 = random.randrange(5, 26, 2)
    kernel_1 = (kernel_1, kernel_1)
    kernel_2 = random.randrange(5, 26, 2)
    kernel_2 = (kernel_2, kernel_2)

    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured[mask_blured < 1] = 0

    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5, 46))
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured = cv2.resize(mask_blured, (W, H))
    return mask_blured.reshape((mask_blured.shape + (1,)))


def get_alpha_blend_mask(mask):
    kernel_list = [(11, 11), (9, 9), (7, 7), (5, 5), (3, 3)]
    blend_list = [0.25, 0.5, 0.75]
    kernel_idxs = random.choices(range(len(kernel_list)), k=2)
    blend_ratio = blend_list[random.sample(range(len(blend_list)), 1)[0]]
    mask_blured = cv2.GaussianBlur(mask, kernel_list[0], 0)
    mask_blured[mask_blured < mask_blured.max()] = 0
    mask_blured[mask_blured > 0] = 1
    mask_blured = cv2.GaussianBlur(mask_blured, kernel_list[kernel_idxs[1]], 0)
    mask_blured = mask_blured / (mask_blured.max())
    return mask_blured.reshape((mask_blured.shape + (1,)))


def IoUfrom2bboxes(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def crop_face(img, landmark=None, bbox=None, margin=False, crop_by_bbox=True, abs_coord=False, only_img=False,
              phase='train'):
    assert phase in ['train', 'val', 'test']

    # crop face------------------------------------------
    H, W = len(img), len(img[0])

    assert landmark is not None or bbox is not None

    if crop_by_bbox:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 4  # 0#np.random.rand()*(w/8)
        w1_margin = w / 4
        h0_margin = h / 4  # 0#np.random.rand()*(h/5)
        h1_margin = h / 4
    else:
        x0, y0 = landmark[:68, 0].min(), landmark[:68, 1].min()
        x1, y1 = landmark[:68, 0].max(), landmark[:68, 1].max()
        w = x1 - x0
        h = y1 - y0
        w0_margin = w / 8  # 0#np.random.rand()*(w/8)
        w1_margin = w / 8
        h0_margin = h / 2  # 0#np.random.rand()*(h/5)
        h1_margin = h / 5

    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == 'train':
        w0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
        w1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
        h0_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
        h1_margin *= (np.random.rand() * 0.6 + 0.2)  # np.random.rand()
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5

    y0_new = max(0, int(y0 - h0_margin))
    y1_new = min(H, int(y1 + h1_margin) + 1)
    x0_new = max(0, int(x0 - w0_margin))
    x1_new = min(W, int(x1 + w1_margin) + 1)

    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p, q) in enumerate(landmark):
            landmark_cropped[i] = [p - x0_new, q - y0_new]
    else:
        landmark_cropped = None
    if bbox is not None:
        bbox_cropped=np.zeros_like(bbox)
        for i,(p,q) in enumerate(bbox):
            bbox_cropped[i] = [p-x0_new,q-y0_new]
    else:
        bbox_cropped = None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped, landmark_cropped, bbox_cropped, (
            y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1), y0_new, y1_new, x0_new, x1_new
    else:
        return img_cropped, landmark_cropped, bbox_cropped, (y0 - y0_new, x0 - x0_new, y1_new - y1, x1_new - x1)
