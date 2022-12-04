import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import gc
import numpy as np
import sys
import torch
from torch import nn
from typing import List
import glob
import faiss
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from numpy.linalg import norm

import argparse
from datasets import Pair
from builder import MoCo
from extraction import FeatureExtractor
from encoder import Net
from helper import mean_error_IOD

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser


parser = argparse.ArgumentParser()
add_dict_to_argparser(parser, model_and_diffusion_defaults())
command = "--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
args = parser.parse_args(command.split(" "))

opts = {
    "model_type": "ddpm",
    "dim": [256, 256, 8448],
    "steps": [150],
    "blocks": [5,6,7,8,12],
    "model_path": "checkpoints/ddpm/ffhq.pt",
    "input_activations": False,
    "upsample_mode":"bilinear",
}
opts.update(vars(args))
opts['image_size'] = opts['dim'][0]
feature_extractor = FeatureExtractor(**opts)

model = MoCo(Net, K=81920)
checkpoint = torch.load("checkpoints/99.pth.tar")
model.load_state_dict(checkpoint["state_dict"])
model = model.cuda().eval()

training = []
with open("../SCOPS/data/CelebA/MAFL/training.txt") as f:
    for line in f:
        training.append(line.strip())
testing = []
with open("../SCOPS/data/CelebA/MAFL/testing.txt") as f:
    for line in f:
        testing.append(line.strip())

np.random.seed(0)
sample_training = sorted(np.random.choice(training, (200,), replace=False))
sample_training = ["datasets/celeba/img_align_celeba/" + x for x in sample_training]
sample_testing = sorted(np.random.choice(testing, (2000,), replace=False))
sample_testing = ["datasets/celeba/img_align_celeba/" + x for x in sample_testing]

train_ds = Pair(opts, sample_training, feature_extractor, train=False)
test_ds = Pair(opts, sample_testing, feature_extractor, train=False)

ann = {}
with open("datasets/celeba/list_landmarks_align_celeba.txt") as f:
    i = 0
    for line in f:
        if i < 2:
            i += 1
            continue
        line = line.strip().split()
        name = "datasets/celeba/img_align_celeba/" + line[0]
        pts = [int(x) for x in line[1:]]
        pts = np.array(pts).reshape(5, 2)

        # w, h, _ = np.array(Image.open(name)).shape
        # pts[:, 0] = pts[:, 0] / h
        # pts[:, 1] = pts[:, 1] / w
        ann[name] = pts

train_feats = []
for k in tqdm(range(len(train_ds))):
    x, _ = train_ds[k]
    x = model.encoder_q.net(x.unsqueeze(0).cuda()).detach().cpu().numpy()
    train_feats.append(x)

test_feats = []
for k in tqdm(range(len(test_ds))):
    x, _ = test_ds[k]
    x = model.encoder_q.net(x.unsqueeze(0).cuda().detach().cpu().numpy())
    test_feats.append(x)

dim = 128

train_feats = np.stack(train_feats, axis=0)
feats = train_feats + test_feats
feats = np.stack(feats, axis=0)

train_feats = train_feats.transpose(0, 2, 3, 1).reshape(-1, dim)
kmeans = faiss.Kmeans(dim, 8, niter=200, gpu=True, spherical=False)
kmeans.train(train_feats)
label = kmeans.assign(feats)[1].reshape(2000, 256, 256)

centers = []
for i in range(200):
    center = []
    curr_label = label[i]
    for j in range(8):
        x, y = np.where(curr_label == j)
        x = np.mean(x)
        y = np.mean(y)
        center.append([x, y])
    centers.append(np.array(center))

train_gt = []
train_pred = []
for i in range(len(sample_training)):
    img = Image.open(sample_training[i])
    w, h, _ = np.array(img).shape
    train_pred.append(np.array([centers[i][:, 1], centers[i][:, 0]]))
    gt = ann[sample_training[i]]
    train_gt.append(np.array([gt[:, 0] / h * 256, gt[:, 1] / w * 256]))
train_pred = np.array(train_pred)
train_gt = np.array(train_gt)

test_gt = []
test_pred = []
for i in range(len(sample_testing)):
    img = Image.open(sample_testing[i])
    w, h, _ = np.array(img).shape
    test_pred.append(np.array([centers[i + 200][:, 1], centers[i + 200][:, 0]]))
    gt = ann[sample_testing[i]]
    test_gt.append(np.array([gt[:, 0] / h * 256, gt[:, 1] / w * 256]))
test_pred = np.array(test_pred)
test_gt = np.array(test_gt)

train_pred[np.isnan(train_pred)] = 0
test_pred[np.isnan(test_pred)] = 0

lr = LinearRegression(fit_intercept=False)
lr.fit(train_pred.reshape(-1, 16), train_gt.reshape(-1, 10))

test_gt_pred = lr.predict(test_pred.reshape(-1, 16))

print(mean_error_IOD(test_gt_pred.reshape(2000, 5, 2), test_gt.reshape(2000, 5, 2)))



