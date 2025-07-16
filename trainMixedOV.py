from openvino.runtime import Core
import cv2
import glob

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
# from datasets.mvtec import MVTecDataset
from datasets.mvtec import MVTecDataset
from utils.functions import (
    cal_anomaly_maps,
    cal_anomaly_maps_RnetEffNet
)
from utils.visualization import plt_fig

from models.studentEffNet import studentEffNet
from models.resnet_reduced_backbone import reduce_student18
from trainDistillation import AnomalyDistillation
from models.teacherTimm import teacherTimm

class MixedTeacherOV:
    def __init__(self, args):
        self.device = "CPU"
        self.data_path = args.data_path
        self.obj = args.obj
        self.img_resize = args.img_resize
        self.img_cropsize = args.img_cropsize
        self.vis = args.vis
        self.img_dir = args.img_dir
        self.KD_effnetPath = args.KD_effnetPath
        self.KD_resnetPath = args.KD_resnetPath

        self.core = Core()

        self.model_t_effNet = self.core.compile_model("teacher_effnet.xml", self.device)
        self.model_t_resNet = self.core.compile_model("teacher_resnet.xml", self.device)
        self.model_KD_effNet = self.core.compile_model(os.path.join(self.KD_effnetPath, "model_s.xml"), self.device)
        self.model_KD_resNet = self.core.compile_model(os.path.join(self.KD_resnetPath, "model_s.xml"), self.device)

        self.infer_t_effNet = self.model_t_effNet.create_infer_request()
        self.infer_t_resNet = self.model_t_resNet.create_infer_request()
        self.infer_s_effNet = self.model_KD_effNet.create_infer_request()
        self.infer_s_resNet = self.model_KD_resNet.create_infer_request()

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.img_resize, self.img_resize))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, H, W)
        return img

    def test(self):
        image_paths = sorted(glob.glob(os.path.join(self.data_path, self.obj, "test", "*", "*.png")))
        scores = []
        test_imgs = []
        gt_list = []
        gt_mask_list = []

        print(f"Testing on {len(image_paths)} images")

        for img_path in tqdm(image_paths):
            img = cv2.imread(img_path)
            test_imgs.append(img)
            input_tensor = self.preprocess(img_path)

            # Forward through all 4 models
            fs_res = self.infer_s_resNet.infer({0: input_tensor})
            ft_res = self.infer_t_resNet.infer({0: input_tensor})
            fs_eff = self.infer_s_effNet.infer({0: input_tensor})
            ft_eff = self.infer_t_effNet.infer({0: input_tensor})

            # Anomaly map calculation (remains the same)
            score = cal_anomaly_maps_RnetEffNet(fs_res, ft_res, fs_eff, ft_eff, self.img_cropsize)
            scores.append(score)

            # Ground truth placeholder (update if needed)
            if "good" in img_path:
                gt_list.append(0)
                gt_mask_list.append(np.zeros((self.img_cropsize, self.img_cropsize)))
            else:
                gt_list.append(1)
                mask_path = img_path.replace("test", "ground_truth").replace(".png", "_mask.png")
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (self.img_cropsize, self.img_cropsize)) / 255
                gt_mask_list.append(mask)

        scores = np.asarray(scores)
        img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
        gt_list = np.asarray(gt_list)

        # Normalize
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        img_roc_auc = roc_auc_score(gt_list, img_scores)

        # Plot
        plt_fig(
            test_imgs,
            scores,
            img_scores,
            gt_mask_list,
            0.1,
            0.1,
            self.img_dir,
            self.obj
        )
        print(self.obj + " image ROCAUC: %.3f" % (img_roc_auc))

