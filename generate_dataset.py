import argparse
import json
import math
import os
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from foreground_segmentor import ISNetDIS
from image_gen_models.img_gen_model import IMG_GEN_MODELS, Generator
from models.model import BACKBONES, FeatureExtractor
from object_data import OBJECT_DATA, get_object_dicts
from utils import torch_seed


def get_argparser():
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument(
        "--img-gen-model",
        type=IMG_GEN_MODELS,
        default=str(IMG_GEN_MODELS.FLUX),
        choices=list(IMG_GEN_MODELS),
        help="Model name",
    )
    parser.add_argument(
        "--object-data",
        type=OBJECT_DATA,
        default=OBJECT_DATA.DEFAULT,
        choices=list(OBJECT_DATA),
        help="Model name",
    )
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "-n", "--n-img", type=int, default=1, help="Number of Generated Img"
    )
    parser.add_argument("--out-path", type=str, default="./synthetic_dataset/")
    parser.add_argument("-s", "--seed", type=int, default=1, help="Seed")
    parser.add_argument("--filter-model", type=BACKBONES, default=BACKBONES.DINOV3)
    parser.add_argument(
        "--filter-thr",
        type=float,
        default=0.25,
        help="Threshold for filtering image pairs without anomalies and to generate ground truth masks",
    )  # Use 0.3 for DINOv3 and DINOv2, other models were not tested
    parser.add_argument(
        "--mode", default=MODE.GENERATE, type=MODE, choices=list(MODE), help="Mode"
    )
    args = parser.parse_args()
    return args


class MODE(Enum):
    GENERATE = "generate"
    GENERATE_ANOM = "generate_anom"
    FILTER_ANOM = "filter"

    def __str__(self):
        return self.value


def get_background(img: Image.Image, bg_model: torch.nn.Module) -> torch.tensor:
    img = (torchvision.transforms.ToTensor()(img) - 0.5) * 2
    img = img.to("cuda")
    img = img.unsqueeze(0)
    bg = bg_model(img)

    result = torch.squeeze(
        F.upsample(bg[0][0], (img.shape[-2], img.shape[-1]), mode="bilinear"), 0
    )
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result - mi) / (ma - mi)
    result = torch.where(result > 0.5, 1.0, 0.0)

    return result.squeeze()


def generate_box(mask: torch.Tensor):
    anom_region = torch.zeros_like(mask)
    wx, wy = torch.randint(50, 300, (1,)).item(), torch.randint(50, 300, (1,)).item()
    coords = torch.nonzero(mask, as_tuple=False)

    idx = torch.randint(len(coords), (1,))
    sampled_coord = coords[idx].squeeze(0)
    sx = max(0, sampled_coord[0] - wx // 2)
    sy = max(0, sampled_coord[1] - wy // 2)
    ex = min(mask.shape[0], sampled_coord[0] + wx // 2)
    ey = min(mask.shape[0], sampled_coord[1] + wy // 2)
    anom_region[sx:ex, sy:ey] = 1
    return anom_region


def generate_mask(
    img1: Image.Image,
    img2: Image.Image,
    filter_processor: torch.nn.Module,
    filter_model: torch.nn.Module,
    upsampler: torch.nn.Module,
    filter_thr: float,
    box=None,
):
    img1 = filter_processor(img1).unsqueeze(0)
    img1 = img1.to("cuda")
    img2 = filter_processor(img2).unsqueeze(0)
    img2 = img2.to("cuda")

    _, x1 = filter_model(img1)
    _, x2 = filter_model(img2)

    B, D, C = x1.shape
    x1 = x1.permute(0, 2, 1).reshape(B, C, int(math.sqrt(D)), int(math.sqrt(D)))
    x2 = x2.permute(0, 2, 1).reshape(B, C, int(math.sqrt(D)), int(math.sqrt(D)))
    x1 = upsampler(x1)
    x2 = upsampler(x2)

    dot = (x1 * x2).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    x1_norm = x1.norm(p=2, dim=1, keepdim=True)  # [B, 1, H, W]
    x2_norm = x2.norm(p=2, dim=1, keepdim=True)  # [B, 1, H, W]

    eps = 1e-8
    cosine_sim = dot / (x1_norm * x2_norm + eps)  # [B, 1, H, W]
    cosine_sim = 1.0 - cosine_sim
    mask = torch.where(cosine_sim > filter_thr, 1, 0).squeeze()

    H = mask.shape[-1]
    trf = transforms.Compose([transforms.Resize((H, H)), transforms.ToTensor()])
    box = trf(box).squeeze()
    box = torch.where(box > 0.0, 1, 0).to("cuda")
    mask = mask * box

    is_ok = torch.any(mask == 1)
    mask = (255.0 * mask).cpu().byte().numpy()
    mask = Image.fromarray(mask)
    return mask, is_ok


if __name__ == "__main__":

    args = get_argparser()

    object_list, damage_dict, backgrounds = get_object_dicts(args.object_data)
    torch_seed(args.seed)

    if args.mode == MODE.GENERATE:
        generator = Generator(args.img_gen_model, inpaint=False)
        width, height = args.image_size, args.image_size

        cnt = 0
        while cnt < args.n_img:
            object_str = object_list[torch.randint(0, len(object_list), (1,)).item()]
            background_text = backgrounds[
                torch.randint(0, len(backgrounds), (1,)).item()
            ]
            seed = torch.randint(0, 10000, (1,)).item()
            if "Texture" in object_str:
                text = f"A close-up photo of a {object_str} for industrial visual inspection. Top down view. Centered."
            else:
                text = f"A close-up photo of a {object_str} for industrial visual inspection. Top down view. Centered. {background_text} background."
            with torch.no_grad():
                img, pth = generator(
                    text,
                    width=width,
                    height=height,
                    seed=seed,
                    save_path=f"{args.out_path}/pile/good/{cnt:04d}.png",
                )

            os.makedirs(os.path.dirname(pth), exist_ok=True)
            img.save(pth, format="png")
            txt_pth = pth.replace("png", "json")
            txt_json = {
                "object": object_str,
                "background": background_text,
                "seed": str(seed),
            }
            with open(txt_pth, "w") as f:
                json.dump(txt_json, f)
            cnt += 1
    elif args.mode == MODE.GENERATE_ANOM:

        bg_generator = ISNetDIS()
        bg_generator.load_state_dict(
            torch.load("./pretrained_models/isnet-general-use.pth")
        )
        bg_generator.to("cuda")
        cnt = 0
        # cnt = args.n_img

        while cnt < args.n_img:
            img_pth = f"{args.out_path}/pile/good/{cnt:04d}.png"
            json_pth = f"{args.out_path}/pile/good/{cnt:04d}.json"
            img = Image.open(img_pth)
            with open(json_pth, "r") as f:
                object_json = json.load(f)
            object_str = object_json["object"]

            bg = get_background(img, bg_generator)
            if "Texture" in object_str:
                bg = torch.ones_like(bg)

            bg_pth = f"{args.out_path}/pile/support_bg/{cnt:04d}.png"

            os.makedirs(os.path.dirname(bg_pth), exist_ok=True)
            bg = torchvision.transforms.transforms.F.to_pil_image(bg)
            bg.save(bg_pth, format="png")

            cnt += 1

        bg_generator.cpu()
        torch.cuda.empty_cache()

        generator = Generator(args.img_gen_model, inpaint=True)
        steps = 20
        width, height = args.image_size, args.image_size

        cnt = 0
        while cnt < args.n_img:
            img_pth = f"{args.out_path}/pile/good/{cnt:04d}.png"
            bg_pth = f"{args.out_path}/pile/support_bg/{cnt:04d}.png"
            json_pth = f"{args.out_path}/pile/good/{cnt:04d}.json"

            img = Image.open(img_pth).convert("RGB")
            bg = Image.open(bg_pth).convert("L")
            bg = torchvision.transforms.transforms.F.to_tensor(bg).squeeze(0)

            anom_region = generate_box(bg)

            with open(json_pth, "r") as f:
                object_json = json.load(f)
            object_str = object_json["object"]
            background_text = object_json["background"]
            seed = torch.randint(0, 10000, (1,)).item()  # int(object_json["seed"])
            anomaly_text = damage_dict[object_str][
                torch.randint(0, len(damage_dict[object_str]), (1,)).item()
            ]
            if "Texture" in object_str:
                text = f"A close-up photo of a {anomaly_text} {object_str} for industrial visual inspection. Top down view. Centered."
            else:
                text = f"A close-up photo of a {anomaly_text} {object_str} for industrial visual inspection. Top down view. Centered. {background_text} background."
            with torch.no_grad():
                img, pth = generator(
                    text,
                    width=width,
                    height=height,
                    seed=seed,
                    init_image=img,
                    bg_mask=anom_region,
                    save_path=f"{args.out_path}/pile/bad/{cnt:04d}.png",
                )

            os.makedirs(os.path.dirname(pth), exist_ok=True)
            img.save(pth, format="png")

            support_pth_anom_region = (
                f"{args.out_path}/pile/support_anom_region/{cnt:04d}.png"
            )
            os.makedirs(os.path.dirname(support_pth_anom_region), exist_ok=True)
            anom_region = (255.0 * anom_region).cpu().byte().numpy()
            anom_region = Image.fromarray(anom_region)
            anom_region.save(support_pth_anom_region, format="png")
            cnt += 1

    elif args.mode == MODE.FILTER_ANOM:
        filter_model = FeatureExtractor(args.filter_model, args.image_size).model
        filter_processor = filter_model.get_img_transform()
        filter_model.to("cuda")
        upsampler = transforms.Compose(
            [
                transforms.Resize(
                    (args.image_size, args.image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            ]
        )
        filter_thr = args.filter_thr

        cnt = 0
        cnt_out = 0
        while cnt < args.n_img:
            img_1_pth = f"{args.out_path}/pile/good/{cnt:04d}.png"
            img_2_pth = f"{args.out_path}/pile/bad/{cnt:04d}.png"
            box_pth = f"{args.out_path}/pile/support_anom_region/{cnt:04d}.png"
            img = Image.open(img_1_pth)
            img2 = Image.open(img_2_pth)
            box = Image.open(box_pth)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    mask, is_ok = generate_mask(
                        img,
                        img2,
                        filter_processor,
                        filter_model,
                        upsampler,
                        filter_thr,
                        box=box,
                    )

            if is_ok:
                pth = f"{args.out_path}/train/ok/{cnt_out:04d}.png"
                os.makedirs(os.path.dirname(pth), exist_ok=True)
                img.save(pth, format="png")
                pth2 = f"{args.out_path}/train/bad/{cnt_out:04d}.png"
                os.makedirs(os.path.dirname(pth2), exist_ok=True)
                img2.save(pth2, format="png")
                mask_pth = f"{args.out_path}/ground_truth/bad/{cnt_out:04d}.png"
                os.makedirs(os.path.dirname(mask_pth), exist_ok=True)
                mask.save(mask_pth, format="png")
                cnt_out += 1
            else:
                print(f"Skipping image {cnt}")
            cnt += 1
