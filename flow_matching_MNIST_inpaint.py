# TODO
# - [x] OT coupling before sampling mini-batch, for unpaired translation
# - [ ] Using same flow matching loss as I2SB eq12.
# - [x] resized MNIST, step=1 training
# - [ ] model is not robust for different num_channles (eg. ERROR at num_channel=128, num_res_block=2)
# - [ ] need implementation for multi-resolution input simultaneously, and calculate summed FM loss

#!/usr/bin/env python
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import random
import os

#from sklearn.datasets import make_moons, make_circles
from torch import Tensor
from tqdm import tqdm
from typing import *
from torch.utils.data import Dataset, DataLoader

#from zuko.utils import odeint
from torchvision.datasets import MNIST, EMNIST
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage, Resize
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torchdyn.core import NeuralODE
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.multiprocessing import spawn
from prefetch_generator import BackgroundGenerator
import wandb


from torchcfm.models.unet import ProUNetModel
from torchcfm.conditional_flow_matching import *
from build_mask import build_inpaint_center

def seed_everything(seed:int=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2
def count_params(model):
    num = sum(para.data.nelement() for para in model.parameters())
    count = num / 1024**2
    print(f"Model num params: {count=:.2f} M")

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def build_datasets(step:int=0):
    # TODO: change resolution online during training
    if step == 0:
        img_resize = IMG_SIZE / 2
        return MNIST("./data", train=True, transform=Compose([ToTensor(), Normalize((.5, ), (.5,)), Resize(16)]), download=True)
    elif step == 1:
        img_resize = IMG_SIZE  
        return MNIST("./data", train=True, transform=Compose([ToTensor(), Normalize((.5, ), (.5,)), Resize(32)]), download=True)
    else:
        raise ValueError("Not providing progressive step")

def init_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1234'
    init_process_group("nccl", rank = rank, world_size=world_size)

def sample_batch(rank, loader, corrupt_method):
    clean_img, y = next(loader)
    with torch.no_grad():
        corrupt_img, mask = corrupt_method(clean_img.to(rank))

    y  = y.detach().to(rank)
    x0 = clean_img.detach().to(rank)
    x1 = corrupt_img.detach().to(rank)
    if mask is not None:
        mask = mask.detach().to(rank)
        x1 = (1. - mask) * x1 + mask * torch.randn_like(x1)
    cond = None
    assert x0.shape == x1.shape
    return x0, x1, mask, y, cond

def build_loader(rank, world_size, dataset, batch_size):
    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size,rank = rank)
    dataloader = DataLoaderX(dataset, batch_size = batch_size, sampler = data_sampler, drop_last=True)
    #return dataloader
    while True:
        yield from dataloader


def main(rank, world_size, **kwargs):
    seed_everything(2024)
    init_process(rank, world_size)
    LOG_FLAG = True
    global IMG_SIZE 
    IMG_SIZE = 32

    if rank==0 and LOG_FLAG:
        wandb.init(project='FM_inpaint')
    corrupt_method_16 = build_inpaint_center(IMG_SIZE//2, rank)
    corrupt_method_32 = build_inpaint_center(IMG_SIZE, rank)

    EPOCH = 100
    batch_size= 256

    model_arch_dict= {
        'learn_sigma':False,
        'use_checkpoint': False,
        'num_heads': 1,
        'num_head_channels': -1,
        'num_heads_upsample': -1,
        'dropout': 0,
        'resblock_updown': False,
        'use_fp16': False,
    }

    #model = ProUNetModel(dim=(1, IMG_SIZE, IMG_SIZE), num_channels=128, num_res_blocks=2, **model_arch_dict).to(rank) 
    model = ProUNetModel(dim=(1, IMG_SIZE, IMG_SIZE), num_channels=32, num_res_blocks=1, **model_arch_dict).to(rank) 
    node = NeuralODE(model, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True) 

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    #---------source dataset----------#
    data_src_16 = build_datasets(0)
    data_src_32 = build_datasets(1)
    dataloader_16 = build_loader(rank, world_size, data_src_16, batch_size)
    dataloader_32 = build_loader(rank, world_size, data_src_32, batch_size)

    #dataloader_src = iter(dataloader)
    dataloader_src_16 = dataloader_16
    dataloader_src_32 = dataloader_32
    model.train()
    for epoch in tqdm(range(EPOCH), colour='GREEN'): 
        loss_value = 0
        ITERS = len(data_src_16) // batch_size 

        # ---------- step configs ---------- # 
        step = 0 if epoch < EPOCH // 2 else 1
        dataloader_src = dataloader_src_16 if step == 0 else dataloader_src_32
        corrupt_method = corrupt_method_16 if step == 0 else corrupt_method_32
        img_size = 16 if step == 0 else 16* step*2

        for iters in tqdm(range(ITERS), desc = f"{step=}"):
            optimizer.zero_grad()
            x0, x1, mask, y, cond = sample_batch(rank, dataloader_src, corrupt_method)
            """
            if rank==0 and LOG_FLAG:
                with torch.no_grad():
                    clean_images = wandb.Image(x0, caption='clean')
                    corrupt_images = wandb.Image(x1, caption='corrupt')
                    wandb.log({"x1": clean_images, "x0": corrupt_images})
            """
            t, xt, ut = FM.sample_location_and_conditional_flow(x1, x0) # x0: clean, x1: paint
            vt = model(t, xt, step=step) # t: [256]; xt: [256,1,28,28]
            loss_v = torch.mean((vt - ut)**2)
            loss_v.backward()
            loss_value += loss_v.item()
            optimizer.step()

        loss_value /= ITERS 
        with torch.no_grad():
            print(f"{loss_value=:.4f}")


    # -------- sampling ----------#

        if rank==0:
            with torch.no_grad():
                x0, x1, mask, y, cond = sample_batch(rank, dataloader_src, corrupt_method)
                traj = node.trajectory(x1, t_span = torch.linspace(0,1,2).to(rank))
                grid_source = make_grid(x1[:100].cpu().view(-1,1,img_size,img_size).clip(-1,1), value_range=(-1,1), padding=0, nrow=10)
                grid_dist = make_grid(x0[:100].cpu().view(-1,1,img_size,img_size).clip(-1,1), value_range=(-1,1), padding=0, nrow=10)
                grid = make_grid(
                    traj[-1, :100].view(-1,1,img_size,img_size).clip(-1,1), value_range=(-1,1), padding=0, nrow=10
                )

                corrupt_img = ToPILImage()(grid_source)
                clean_img = ToPILImage()(grid_dist)
                clean_image_infer_pil = ToPILImage()(grid)

                clean_img_infer = wandb.Image(grid, caption='result')
                clean_images= wandb.Image(grid_source, caption='source')
                corrupt_images= wandb.Image(grid_dist, caption='corrupt')
                if LOG_FLAG: wandb.log({"x1_infer": clean_img_infer, "x0": clean_images, "x1": corrupt_images})             
                clean_image_infer_pil.save("x0_infer.png")
                clean_img.save("x0.png")
                corrupt_img.save("x1.png")
    barrier()
    destroy_process_group()


def spawn_fn(fn):
    world_size = 4
    print(f'\033[93m world size per node {world_size=}\033[0m')
    spawn(fn,
          args=(world_size,),# argv[0] is rank
          nprocs = world_size,
          join=True
          )


if __name__ == '__main__':

    spawn_fn(main)

