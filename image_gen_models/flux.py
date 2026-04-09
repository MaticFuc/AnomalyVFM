import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from PIL import Image

from flux.cli import SamplingOptions
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import configs, load_ae, load_clip, load_flow_model, load_t5


def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


class Flux(nn.Module):

    def __init__(self, model_name, device, offload):
        super().__init__()
        self.gen = FluxGenerator(model_name, device, offload)
        self.steps = 20

    def forward(self, txt, **kwargs):
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)

        init_image = kwargs.get("init_image", None)
        bg_mask = kwargs.get("bg_mask", None)

        save_path = kwargs["save_path"]
        seed = kwargs["seed"]

        return self.gen.generate_image(
            width,
            height,
            20,
            4.0,
            seed,
            txt,
            init_image=init_image,
            bg_mask=bg_mask,
            save_path=save_path,
        )


class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool):
        self.device = torch.device(device)
        self.model_name = model_name
        self.is_schnell = model_name == "flux-schnell"
        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,
            device=self.device,
            offload=False,
            is_schnell=self.is_schnell,
        )
        self.offload = offload

    @torch.inference_mode()
    def generate_image(
        self,
        width,
        height,
        num_steps,
        guidance,
        seed,
        prompt,
        init_image=None,
        image2image_strength=0.0,
        add_sampling_metadata=True,
        bg_mask=None,
        save_path=None,
    ):
        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        if self.offload:
            self.ae = self.ae.to("cuda")

        if init_image is not None:
            init_image = (torchvision.transforms.ToTensor()(init_image) - 0.5) * 2
            init_image = init_image.to(self.device)
            init_image = init_image.unsqueeze(0)
            if self.offload:
                self.ae.encoder.to(self.device)
            init_image = self.ae.encode(init_image.to("cuda"))
            if self.offload:
                self.ae = self.ae.cpu()
                torch.cuda.empty_cache()

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=(not self.is_schnell),
        )
        orig_img = None
        bg_mask_down = None
        if bg_mask is not None:
            orig_img = x - init_image
            bg_mask_down = torch.nn.functional.interpolate(
                bg_mask.unsqueeze(0).unsqueeze(0), (x.shape[-1], x.shape[-1])
            )
            bg_mask_down = bg_mask_down.to("cuda").bfloat16()
            orig_img = orig_img.to("cuda").bfloat16()

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        if bg_mask is not None:
            orig_img = rearrange(
                orig_img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
            )
            bg_mask_down = bg_mask_down.repeat(1, 16, 1, 1)
            bg_mask_down = rearrange(
                bg_mask_down, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
            )

        x = denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            guidance=opts.guidance,
            bg_mask=bg_mask_down,
            orig_img=orig_img
        )

        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return img, save_path
