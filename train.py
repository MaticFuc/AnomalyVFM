import argparse
import os
import warnings
from test import test

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from aux_dataset import AuxilaryDataset
from decoder import SimpleDecoder, SimplePredictor
from logger import log_results
from models.model import BACKBONES, FeatureExtractor
from peft_local.peft_func import PeftType
from utils import (OPTIMIZERS, SCHEDULERS, get_optimizer, get_scheduler,
                   torch_seed)

warnings.filterwarnings("ignore", message="invalid value encountered in divide")


def get_args():
    parser = argparse.ArgumentParser("")
    parser.add_argument(
        "-m",
        "--model",
        default=str(BACKBONES.RADIO),
        type=BACKBONES,
        choices=list(BACKBONES),
    )
    parser.add_argument("-as", "--accumulation-steps", default=4, type=int)
    parser.add_argument("-bs", "--batch-size", default=32, type=int)
    parser.add_argument(
        "-wd", "--weight-decay", default=1e-2, type=float
    )  # For MUON it should be 0
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument(
        "--optimizer",
        default=str(OPTIMIZERS.ADAMW),
        type=OPTIMIZERS,
        choices=list(OPTIMIZERS),
    )
    parser.add_argument(
        "--scheduler",
        default=str(SCHEDULERS.NONE),
        type=SCHEDULERS,
        choices=list(SCHEDULERS),
    )
    parser.add_argument(
        "--peft-type", default=PeftType.DORA, type=PeftType, choices=list(PeftType)
    )
    parser.add_argument(
        "--peft-rank", default=64, type=int
    )  # Also works well with smaller ranks (e.g. 8 or 16)

    parser.add_argument("--seed", default=12, type=int)

    parser.add_argument(
        "-d",
        "--data-path",
        default="./synthetic_dataset_flux_filter_dinov3/",
    )
    parser.add_argument(
        "--image-size", default=768, type=int
    )  # For DINOv2 and CLIP we used 672 to get the same ftr size
    parser.add_argument(
        "-o", "--out-path", default="./experiments/radio_200_it/", type=str
    )

    parser.add_argument("-t", "--train-steps", default=200, type=int)
    parser.add_argument("-ts", "--test-steps", default=100, type=int)

    parser.add_argument(
        "--evaluate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--mean-kernel-size", default=5, type=int
    )  # This stabilizes the pixel-level metrics, without this the results are more random from run to run.
    parser.add_argument(
        "-td",
        "--test-datasets",
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
        ],
        nargs="+",
    )  # These are the datasets used in the main paper, more datasets can be added. (Instructions in the datasets/ADDING_NEW_DATASET.md)
    parser.add_argument(
        "-mdi",
        "--medical-test-datasets-img",
        default=["headct", "brainmri", "br35h"],
        nargs="+",
    )
    parser.add_argument(
        "-mdp",
        "--medical-test-datasets-pix",
        default=["isic", "cvc_colondb", "cvc_clinicdb", "kvasir", "endo", "thyro"],
        nargs="+",
    )

    args = parser.parse_args()
    return args


def turn_to_exp(conf_map):
    return 1 + conf_map.exp()


def freeze_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


def main(args):
    torch_seed(args.seed)
    model = FeatureExtractor(args.model, args.image_size).model

    feat_dim = model.feature_dim
    feat_size = args.image_size // model.patch_size

    total_params = sum(p.numel() for p in model.parameters())
    freeze_parameters(model)

    peft_rank = args.peft_rank
    model.add_peft(peft_rank, peft_type=args.peft_type)

    test_datasets = args.test_datasets
    medical_test_datasets_img = args.medical_test_datasets_img
    medical_test_datasets_pix = args.medical_test_datasets_pix

    num_up_layers = 1

    decoder = SimpleDecoder(feat_dim, num_up_layers, 1)

    if args.model in [BACKBONES.RADIO]:  # RADIO has 3 CLS tokens
        predictor = SimplePredictor(feat_dim * 3)
    elif args.model in [BACKBONES.TIPSV2]: # TIPSv2 has 2 CLS tokens
        predictor = SimplePredictor(feat_dim * 2)
    else:
        predictor = SimplePredictor(feat_dim)

    model.cuda()
    decoder.cuda()
    predictor.cuda()

    model.train()
    decoder.train()
    predictor.train()

    opt = get_optimizer(args.optimizer)
    optimizer = opt(
        [
            {"params": model.parameters()},
            {"params": decoder.parameters()},
            {"params": predictor.parameters()},
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer.zero_grad()

    scheduler = get_scheduler(
        args.scheduler, optimizer, num_iterations=args.train_steps
    )

    img_transform = model.get_img_transform()

    mask_transform = T.Compose(
        [
            T.Resize((feat_size * (2**num_up_layers), feat_size * (2**num_up_layers))),
        ]
    )

    dataset = AuxilaryDataset(args.data_path, img_transform, mask_transform)

    batch_size = args.batch_size // args.accumulation_steps
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    infinite_loader = InfiniteDataloader(loader)

    alpha = 0.1
    accumulation_steps = (
        args.accumulation_steps
    )  # Makes effective batch size = batch size * accumulation steps

    img_loss = torchvision.ops.sigmoid_focal_loss
    mask_foc_loss = torchvision.ops.sigmoid_focal_loss
    mask_l1_loss = torch.nn.L1Loss(reduction="none")

    tqdm_obj = tqdm(range(args.train_steps))
    test_num = args.test_steps
    for it in tqdm_obj:
        inner_loop = range(accumulation_steps)
        loss = 0
        for _, sample in zip(inner_loop, infinite_loader):
            image = sample["image"].cuda()
            mask_gt = sample["mask"].cuda()
            score_gt = sample["is_anom"].cuda()
            summary, ftrs = model(image)

            ftrs = ftrs.permute(0, 2, 1)
            ftrs = ftrs.reshape(-1, feat_dim, feat_size, feat_size)

            mask, c = decoder(ftrs)

            score = predictor(summary).squeeze(1)
            l_img = img_loss(score, score_gt, reduction="mean")

            # The clipping makes the traning extremely more stable, it should be kept for general use
            # It will be detailed in a follow-up extension paper
            inner_mask_loss = 5 * mask_foc_loss(
                mask, mask_gt, reduction="none"
            ) + mask_l1_loss(mask.sigmoid().clip(0.1, 0.9), mask_gt)

            c = turn_to_exp(c)
            l_mask_1 = c * inner_mask_loss
            l_mask_2 = alpha * c.log()
            l_mask = l_mask_1 - l_mask_2

            l_mask = l_mask.mean()

            loss = l_img + l_mask
            loss /= accumulation_steps
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if it % 1 == 0:
            tqdm_obj.set_description(
                "Current loss: {:.4f}, img loss {:.4f}, mask loss {:.4f}".format(
                    loss.item(), l_img.item(), l_mask.item()
                )
            )

        if args.evaluate and it > 0 and (it + 1) % test_num == 0:

            model.eval()
            decoder.eval()
            predictor.eval()
            ind_stats = test(
                model,
                decoder,
                predictor,
                test_datasets,
                img_transform,
                mask_transform,
                args.out_path,
                model_name=args.model,
                feat_size=feat_size,
                kernel=args.mean_kernel_size,
            )

            ind_df = {"Dataset": ["All"], "Iteration": [it]}
            ind_df.update(ind_stats)
            log_results(ind_df, args.out_path, file="train.csv")

            if medical_test_datasets_img is not None:
                med_img_stats = test(
                    model,
                    decoder,
                    predictor,
                    medical_test_datasets_img,
                    img_transform,
                    mask_transform,
                    args.out_path,
                    model_name=args.model,
                    feat_size=feat_size,
                    kernel=args.mean_kernel_size,
                )
                med_img_df = {"Dataset": ["All"], "Iteration": [it]}
                med_img_df.update(med_img_stats)
                log_results(med_img_df, args.out_path, file="train_med_img.csv")

            if medical_test_datasets_pix is not None:
                med_pix_stats = test(
                    model,
                    decoder,
                    predictor,
                    medical_test_datasets_pix,
                    img_transform,
                    mask_transform,
                    args.out_path,
                    model_name=args.model,
                    feat_size=feat_size,
                    kernel=args.mean_kernel_size,
                )
                med_pix_df = {"Dataset": ["All"], "Iteration": [it]}
                med_pix_df.update(med_pix_stats)
                log_results(med_pix_df, args.out_path, file="train_med_pix.csv")

            model.train()
            decoder.train()
            predictor.train()

    os.makedirs(args.out_path, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
        },
        f"{args.out_path}/model.pkl",
    )


if __name__ == "__main__":

    args = get_args()
    main(args)
