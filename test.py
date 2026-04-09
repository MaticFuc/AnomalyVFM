import argparse
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from adeval import EvalAccumulatorCuda
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


class RETURN_VALUES(Enum):
    IMG_ROC = "AUROC-IMG"
    IMG_F1 = "F1-IMG"
    IMG_AP = "AP-IMG"
    PIX_ROC = "AUROC-PIXEL"
    PIX_F1 = "F1-PIXEL"
    PIX_AP = "AP-PIXEL"
    PIX_AUPRO = "AUPRO-0.3"

    def __str__(self):
        return self.value

    @classmethod
    def image_metrics(cls):
        return [cls.IMG_ROC, cls.IMG_F1, cls.IMG_AP]

    @classmethod
    def pixel_metrics(cls):
        return [cls.PIX_ROC, cls.PIX_F1, cls.PIX_AP, cls.PIX_AUPRO]


def get_dataset(name):
    for d in DATASET_RESOURCES:
        if d.name == name:
            return d.class_names, d.init_class, d.path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=[
            "mvtec_ad",
            "visa",
            "mpdd",
            "btad",
            "real_iad",
            "ksdd",
            "ksdd2",
            "dagm",
            "dtd",
            "isic",
            "cvc_colondb",
            "cvc_clinicdb",
            "endo",
            "kvasir",
            "thyro",
        ],
    )

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


def save_predictions_with_paths(tensors, original_paths, out_base_path, suffix=""):
    base_out = Path(out_base_path)

    for tensor, path_str in zip(tensors, original_paths):
        p = Path(path_str)

        out_file = base_out / p.relative_to(p.anchor)
        if suffix != "":
            out_file = out_file.with_stem(f"{out_file.stem}_{suffix}")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        img_np = tensor.detach().cpu().squeeze().numpy()
        img_np = (img_np * 255.0).astype(np.uint8)

        if img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW to HWC
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(out_file), img_np)


def test_category(
    model,
    decoder,
    predictor,
    ds,
    save_imgs=False,
    out_path="./visual_results/",
    kernel=None,
    mask_size=None,
    model_name=BACKBONES.RADIO,
    feat_size=48,
    batch_size=16,
    fast=True,
):
    loader = DataLoader(ds, batch_size, num_workers=4)
    ptr = 0

    gts_scores = torch.zeros((len(ds))).cuda()
    pred_scores = torch.zeros((len(ds))).cuda()

    dtype = torch.float16 if fast else torch.float32

    if mask_size is None:
        gts = torch.zeros((len(ds), 192, 192)).cuda()
        preds = torch.zeros((len(ds), 192, 192)).cuda()
    else:
        gts = torch.zeros((len(ds), mask_size, mask_size)).cuda()
        preds = torch.zeros((len(ds), mask_size, mask_size)).cuda()

    if kernel is not None:
        k = nn.AvgPool2d((kernel, kernel), 1, kernel // 2)

    with tqdm(total=len(ds)) as pbar:
        for sample in loader:
            image = sample["image"].cuda()
            mask_gt = sample["mask"].cuda()
            score_gt = sample["is_anom"].cuda()
            paths = sample["path"]

            batch_sz = image.size(0)

            idx = slice(ptr, ptr + batch_sz)

            with torch.amp.autocast("cuda", dtype=dtype):
                with torch.no_grad():
                    summary, ftrs = model(image)

                    ftrs = ftrs.permute(0, 2, 1)
                    ftrs = ftrs.reshape(batch_sz, -1, feat_size, feat_size)

                    mask, c = decoder(ftrs)
                    score = predictor(summary).squeeze().sigmoid()

            mask_gt = mask_gt.squeeze()
            mask = mask.sigmoid()

            gts_scores[idx] = sample["is_anom"].cuda()
            pred_scores[idx] = score

            mask = torch.nn.functional.interpolate(mask, (mask_size, mask_size))
            mask = mask.squeeze()
            if kernel is not None:
                mask = k(mask)

            if save_imgs:
                save_predictions_with_paths(image, paths, out_path, suffix="img")
                save_predictions_with_paths(mask, paths, out_path, suffix="pred")
                save_predictions_with_paths(mask_gt, paths, out_path, suffix="gt")

            gts[idx, :, :] = mask_gt
            preds[idx, :, :] = mask

            ptr += batch_sz
            pbar.update(batch_sz)

    score_min = min(pred_scores).item()
    score_max = max(pred_scores).item()
    anomap_min = preds.min().item()
    anomap_max = preds.max().item()

    accum = EvalAccumulatorCuda(
        score_min,
        score_max,
        anomap_min,
        anomap_max,
        skip_pixel_aupro=False,
        nstrips=200,
    )

    accum.add_anomap_batch(
        preds.float().cuda(non_blocking=True), gts.byte().cuda(non_blocking=True)
    )

    metrics = accum.summary()

    gts = gts.cpu().numpy().ravel()
    preds = preds.cpu().numpy().ravel()

    gts_scores = gts_scores.cpu().numpy()
    pred_scores = pred_scores.cpu().numpy()

    precision, recall, thresholds = precision_recall_curve(gts, preds)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
    f1_pixel = np.max(f1_score)

    precision, recall, thresholds = precision_recall_curve(gts_scores, pred_scores)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
    f1_img = np.max(f1_score)

    roc_img = roc_auc_score(gts_scores, pred_scores)
    ap_img = average_precision_score(gts_scores, pred_scores)

    print(
        f"IMG SCORES: {str(RETURN_VALUES.IMG_ROC)}: {roc_img}, {str(RETURN_VALUES.IMG_F1)}: {f1_img}, {str(RETURN_VALUES.IMG_AP)}: {ap_img}"
    )
    print(
        f"PIXEL SCORES: {str(RETURN_VALUES.PIX_ROC)}: {metrics['p_auroc']}, {str(RETURN_VALUES.PIX_F1)}: {f1_pixel}, {str(RETURN_VALUES.PIX_AP)}: {metrics['p_aupr']}, {str(RETURN_VALUES.PIX_AUPRO)}: {metrics['p_aupro']}"
    )

    ret = {
        str(RETURN_VALUES.IMG_ROC): roc_img,
        str(RETURN_VALUES.IMG_F1): f1_img,
        str(RETURN_VALUES.IMG_AP): ap_img,
        str(RETURN_VALUES.PIX_ROC): metrics["p_auroc"],
        str(RETURN_VALUES.PIX_F1): f1_pixel,
        str(RETURN_VALUES.PIX_AP): metrics["p_aupr"],
        str(RETURN_VALUES.PIX_AUPRO): metrics["p_aupro"],
    }
    return ret


def test(
    model,
    decoder,
    predictor,
    datasets,
    img_transform,
    mask_transform,
    out_path,
    save_images=False,
    kernel=None,
    model_name="radio",
    feat_size=48,
    batch_size=16,
    fast=True,
    logging=True,
):
    metric_names = list([str(r) for r in RETURN_VALUES])
    img_metric_names = list([str(r) for r in RETURN_VALUES.image_metrics()])
    pix_metric_names = list([str(r) for r in RETURN_VALUES.pixel_metrics()])
    avg_stats = {m: 0.0 for m in metric_names}
    only_img_level_ds = ["headct", "brainmri", "br35h"]
    only_pixel_level_ds = [
        "isic",
        "cvc_colondb",
        "cvc_clinicdb",
        "endo",
        "kvasir",
        "thyro",
    ]
    num_img = 0
    num_pixel = 0
    mask_size = mask_transform(torch.rand((1, 1, 1))).shape[-1]

    for d in datasets:
        print(f"Testing {d} dataset")
        os.makedirs(f"{out_path}/", exist_ok=True)
        cls_names, init_class, pth = get_dataset(d)
        ds_avg_stats = {m: 0.0 for m in metric_names}

        for cls in cls_names:
            print(f"Category {cls}")

            ds = init_class(pth, cls, img_transform, mask_transform)
            cls_res = test_category(
                model,
                decoder,
                predictor,
                ds,
                save_imgs=save_images,
                out_path=f"{out_path}/visual/{d}/",
                kernel=kernel,
                mask_size=mask_size,
                model_name=model_name,
                feat_size=feat_size,
                batch_size=batch_size,
                fast=fast,
            )

            for m in metric_names:
                ds_avg_stats[m] += cls_res[m]

            cls_df = {"Dataset": [d], "Category": [cls]}
            cls_df.update(cls_res)
            if logging:
                log_results(cls_df, out_path)

        for m in metric_names:
            ds_avg_stats[m] /= len(cls_names)

        if d not in only_pixel_level_ds:
            for m in img_metric_names:
                avg_stats[m] += ds_avg_stats[m]
            num_img += 1

        if d not in only_img_level_ds:
            for m in pix_metric_names:
                avg_stats[m] += ds_avg_stats[m]
            num_pixel += 1

        ds_df = {"Dataset": [d], "Category": ["Average"]}
        ds_df.update(ds_avg_stats)
        if logging:
            log_results(ds_df, out_path, file="results_dataset.csv")
        print()

    if num_img > 0:
        for m in img_metric_names:
            avg_stats[m] /= num_img
    if num_pixel > 0:
        for m in pix_metric_names:
            avg_stats[m] /= num_pixel

    avg_df = {"Dataset": ["Average"], "Category": ["Average"]}
    avg_df.update(avg_stats)
    if logging:
        log_results(avg_df, out_path, file="results_dataset.csv")
    return avg_stats


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

    mask_transform = T.Compose(
        [
            T.Resize((feat_size * (2**num_up_layers), feat_size * (2**num_up_layers))),
        ]
    )
    test(
        model,
        decoder,
        predictor,
        args.datasets,
        img_transform,
        mask_transform,
        args.out_path,
        save_images=args.save_images,
        kernel=args.mean_kernel_size,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
