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

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import argparse
from datasets import Pair
from builder import MoCo
from extraction import FeatureExtractor
from encoder import Net

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser

def main():
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

    files = sorted(glob.glob("datasets/ffhq/train/*"))
    dataset = Pair(opts, files, feature_extractor)

    model = MoCo(Net, K=81920).cuda()
    opt = torch.optim.Adam(model.parameters(), lr = 1e-3)
    ce = nn.CrossEntropyLoss().cuda()

    for i in tqdm(range(10)):
        for j in range(len(dataset)):
            x1, x2, coord1, coord2, flag1, flag2 = dataset[j]
            x1 = x1.unsqueeze(0)
            x1 = torch.nn.functional.interpolate(x1, (64, 64), mode="bilinear").cuda()
            x2 = x2.unsqueeze(0)
            x2 = torch.nn.functional.interpolate(x2, (64, 64), mode="bilinear").cuda()
            coord1 = torch.Tensor([coord1]).cuda()
            coord2 = torch.Tensor([coord2]).cuda()
            flag1 = torch.Tensor([flag1]).cuda()
            flag2 = torch.Tensor([flag2]).cuda()
            logits, label = model(x1, x2, coord1, coord2, flag1, flag2)
            # print(logits, label)
            loss = ce(logits, label.cuda(1))
            loss.backward()
            opt.step()
            opt.zero_grad()
        print(loss.item())

        save_checkpoint({
                    'epoch': i + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : opt.state_dict(),
            }, filename='checkpoints/%i.pth.tar' %i)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

if __name__ == '__main__':
    main()




