import argparse
import math
import os
import inspect

import torch
from einops import rearrange, repeat
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.v2 as T
import numpy as np

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import clear_umt5_memory, get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface
from rcm.utils.model_utils import load_state_dict

from imaginaire.utils import distributed

from modify_model import (
    select_model, replace_attention, replace_linear_norm, replace_parallel_linear, tensor_kwargs as default_tensor_kwargs
)

torch._dynamo.config.suppress_errors = True

from datetime import datetime

model = "Wan2.2-A14B"
low_noise_model_pth = "model/TurboWan2.2-I2V-A14B-low-720P-quant.pth"
high_noise_model_pth = "model/TurboWan2.2-I2V-A14B-high-720P-quant.pth"
vae_pth = "checkpoints/Wan2.1_VAE.pth"
save_path = f"output/{datetime.now().strftime("%Y%m%d_%H%M%S")}_generated_video.mp4"
ode = True
boundary = 0.9
sla_topk = 0.1
quant_linear = True
default_norm = False
attention_type = "sagesla"

from contextlib import contextmanager
@contextmanager
def memory_tracker(description):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start_allocated = torch.cuda.memory_allocated()
    start_reserved = torch.cuda.memory_reserved()

    log.info(f"{description} - Memory allocated before: {start_allocated / 1e9:.2f} GB", rank0_only=False)
    log.info(f"{description} - Memory reserved before: {start_reserved / 1e9:.2f} GB", rank0_only=False)

    yield

    torch.cuda.empty_cache()
    end_allocated = torch.cuda.memory_allocated()
    end_reserved = torch.cuda.memory_reserved()
    log.info(f"{description} - Memory allocated after: {end_allocated / 1e9:.2f} GB", rank0_only=False)
    log.info(f"{description} - Memory reserved after: {end_reserved / 1e9:.2f} GB", rank0_only=False)
    log.info(f"{description} - Peak memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", rank0_only=False)
    log.info(f"{description} - Peak memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB", rank0_only=False)


def load_model(model_path):
    with torch.device("meta"):
        net = select_model(model)

    if attention_type in ['sla', 'sagesla']:
        net = replace_attention(net, attention_type=attention_type, sla_topk=sla_topk)

    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    replace_linear_norm(net, replace_linear=quant_linear, replace_norm=not default_norm, quantize=False)

    if distributed.is_rank0():
        state_dict = load_state_dict(model_path)
        state_dict_list = [state_dict]
    else:
        state_dict_list = [{}]

    distributed.dist.broadcast_object_list(state_dict_list, src=0)
    distributed.barrier()
    if not distributed.is_rank0():
        state_dict = state_dict_list[0]

    # for name, param in state_dict.items():
    #     if "blocks" in name and ("weight" in name or "scale" in name) and "norm" not in name and "proj_l" not in name:
    #         sz = param.shape[1] // world_size
    #         param = param[:, rank * sz : (rank + 1) * sz].contiguous()
    #         state_dict[name] = param

    net.load_state_dict(state_dict, assign=True)
    del state_dict

    replace_parallel_linear(net, parallel_size=world_size)
    return net

def main():
    distributed.init()
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()

    tensor_kwargs = {
        "device": torch.device(f"cuda:{rank}"),
        "dtype": torch.bfloat16,
    }
    device = tensor_kwargs["device"]

    tokenizer = Wan2pt1VAEInterface(vae_pth=vae_pth)

    data = torch.load("data.pt")
    x = data["x"].to(device)
    ones = data["ones"].to(device)
    t_steps = data["t_steps"].to(device)
    total_steps = data["total_steps"]
    x_B_C_T_H_W = data["x_B_C_T_H_W"]
    crossattn_emb = data["crossattn_emb"].to(device)
    y_B_C_T_H_W = data["y_B_C_T_H_W"].to(device)
    generator = torch.Generator(device=tensor_kwargs["device"])

    with memory_tracker("Sampling"):
        low_noise_model = load_model(low_noise_model_pth).eval().cpu()
        torch.cuda.empty_cache()
        high_noise_model = load_model(high_noise_model_pth).eval().to(device)

        net = high_noise_model
        switched = False
        for i, (t_cur, t_next) in enumerate(tqdm(list(zip(t_steps[:-1], t_steps[1:])), desc="Sampling", total=total_steps)):
            if t_cur.item() < boundary and not switched:
                high_noise_model.cpu()
                torch.cuda.empty_cache()
                low_noise_model.to(device)
                net = low_noise_model
                switched = True
                log.info("Switched to low noise model.")
            with torch.no_grad():
                v_pred = net(x_B_C_T_H_W=x.to(**tensor_kwargs), timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs), crossattn_emb=crossattn_emb, y_B_C_T_H_W=y_B_C_T_H_W).to(
                    torch.float64
                )
                if ode:
                    x = x - (t_cur - t_next) * v_pred
                else:
                    x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                        *x.shape,
                        dtype=torch.float32,
                        device=tensor_kwargs["device"],
                        generator=generator,
                    )
        samples = x.float()
        low_noise_model.cpu()
        torch.cuda.empty_cache()

    if rank == 0:
        log.info("Decoding generated samples...")
        with torch.no_grad():
            video = tokenizer.decode(samples)

        log.success(f"Rank {rank}: Successfully generated videos", rank0_only=False)

        log.info("Gathering results from all ranks...")

        video = video.cpu()
        video = (1.0 + video.clamp(-1, 1)) / 2.0
        
        # Rearrange for saving: [num_samples, 3, F, H, W] -> [3, F, H, num_samples*W]
        # This creates a horizontal grid of all samples
        to_save = rearrange(video, "n c t h w -> c t h (n w)")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        # Save video
        save_image_or_video(to_save, save_path, fps=16)
        
        log.success(f"Successfully saved {len(video)} generated videos to: {save_path}")
        log.info(f"Video shape: {video.shape} (samples, channels, frames, height, width)")

    distributed.barrier()
        

if __name__ == "__main__":
    main()
