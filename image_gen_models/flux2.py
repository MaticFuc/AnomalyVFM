import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from PIL import Image

from flux2.sampling import (batched_prc_img, batched_prc_txt, denoise,
                            denoise_cached, denoise_cfg, encode_image_refs,
                            get_schedule, scatter_ids)
from flux2.util import (FLUX2_MODEL_INFO, load_ae, load_flow_model,
                        load_text_encoder)


def get_models(model_name: str):
    model_info = FLUX2_MODEL_INFO[model_name]
    torch_device = torch.device("cuda")

    text_encoder = load_text_encoder(model_name, device=torch_device)
    if "klein" in model_name:
        mod_and_upsampling_model = load_text_encoder("flux.2-dev")
    else:
        mod_and_upsampling_model = text_encoder

    model = load_flow_model(model_name, torch_device)
    ae = load_ae(model_name)
    model.eval()
    ae.eval()
    text_encoder.eval()
    return model_info, model, ae, text_encoder, mod_and_upsampling_model


class Flux2(nn.Module):

    def __init__(self, model_name, device, offload):
        super().__init__()
        # self.gen = Flux2Generator(model_name, device)
        self.gen = Flux2Generator("flux.2-klein-base-4b", device)
        if "klein" in model_name:
            self.steps = 4
        else:
            self.steps = 50

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
            self.steps,
            4.0,
            seed,
            txt,
            init_image=init_image,
            bg_mask=bg_mask,
            save_path=save_path,
        )


class Flux2Generator:
    def __init__(self, model_name: str, device: str):
        self.device = torch.device(device)
        self.model_name = model_name
        (
            self.model_info,
            self.model,
            self.ae,
            self.text_encoder,
            self.mod_and_upsampling_model,
        ) = get_models(
            model_name,
        )
        self.offload = False

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
        upsampled_prompts = self.mod_and_upsampling_model.upsample_prompt(
            [prompt], None
        )
        prompt = upsampled_prompts[0] if upsampled_prompts else prompt

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

        if self.model_info["guidance_distilled"]:
            ctx = self.text_encoder([prompt]).to(torch.bfloat16)
        else:
            ctx_empty = self.text_encoder([""]).to(torch.bfloat16)
            ctx_prompt = self.text_encoder([prompt]).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
        ctx, ctx_ids = batched_prc_txt(ctx)

        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        randn = torch.randn(
            shape, generator=generator, dtype=torch.bfloat16, device="cuda"
        )
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(num_steps, x.shape[1])
        denoise_fn = denoise
        x = denoise_fn(
            self.model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=4.0,
            init_image=init_image,
            bg_mask=bg_mask,
        )

        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
        x = self.ae.decode(x).float()

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

        return img, save_path
