import argparse
import os
import warnings
from pathlib import Path
from test import save_predictions_with_paths

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from adeval import EvalAccumulatorCuda
from PIL import Image
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import DATASET_RESOURCES
from decoder import SimpleDecoder, SimplePredictor
from logger import log_results
from peft_local.peft_func import PeftType

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

from enum import Enum

from models.model import BACKBONES, FeatureExtractor


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--image-path", default="test.png", type=str)

    parser.add_argument(
        "-m",
        "--model",
        default=str(BACKBONES.RADIO),
        type=BACKBONES,
        choices=list(BACKBONES),
    )
    parser.add_argument(
        "-p",
        "--model-path",
        default="./pretrained_models/anomalyvfm_radio.pkl",
    )
    parser.add_argument(
        "--peft-type", default=PeftType.DORA, type=PeftType, choices=list(PeftType)
    )
    parser.add_argument("--peft-rank", default=64, type=int)

    parser.add_argument(
        "--image-size", default=768, type=int
    )  # For DINOv2 and CLIP we used 672 to get the same ftr size
    parser.add_argument("-s", "--save-images", action="store_true", default=False)
    parser.add_argument(
        "--logging", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("-o", "--out-path", default="./vis_res/")
    parser.add_argument("-k", "--mean-kernel-size", default=5)

    args = parser.parse_args()
    return args


def main(args):

    model = FeatureExtractor(args.model, args.image_size).model
    feat_dim = model.feature_dim
    feat_size = args.image_size // model.patch_size

    peft_rank = args.peft_rank
    model.add_peft(peft_rank, peft_type=args.peft_type)

    num_up_layers = 1
    decoder = SimpleDecoder(feat_dim, num_up_layers, 1)

    if args.model in [BACKBONES.RADIO]:
        predictor = SimplePredictor(3 * feat_dim)
    else:
        predictor = SimplePredictor(feat_dim)

    state_dicts = torch.load(f"{args.model_path}")
    model.load_state_dict(state_dicts["model_state_dict"])
    decoder.load_state_dict(state_dicts["decoder_state_dict"])
    predictor.load_state_dict(state_dicts["predictor_state_dict"])

    model.cuda()
    decoder.cuda()
    predictor.cuda()

    model.eval()
    decoder.eval()
    predictor.eval()

    img_transform = model.get_img_transform()
    k = nn.AvgPool2d(
        (args.mean_kernel_size, args.mean_kernel_size), 1, args.mean_kernel_size // 2
    )

    image = Image.open(args.image_path).convert("RGB")
    image = img_transform(image).unsqueeze(0).cuda()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            summary, ftrs = model(image)
            ftrs = ftrs.permute(0, 2, 1)
            ftrs = ftrs.reshape(1, -1, feat_size, feat_size)

            mask, c = decoder(ftrs)
            score = predictor(summary).squeeze().sigmoid()
    mask = k(mask)
    save_predictions_with_paths(mask.float().sigmoid(), ["./pred.png"], ".", suffix="")


if __name__ == "__main__":
    args = get_args()
    main(args)
