# TODO
# - [ ] OT coupling before sampling mini-batch

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
#from zuko.utils import odeint
from torchvision.datasets import MNIST, EMNIST
from torchvision.transforms import ToTensor, Compose, Normalize, ToPILImage
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from torchdyn.core import NeuralODE
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.multiprocessing import spawn


from torchcfm.models.unet import UNetModel
from torchcfm.conditional_flow_matching import *
from build_mask import build_inpaint_center

def seed_everything():
    seed = torch.seed()
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def log_normal(x: Tensor) -> Tensor:
    return -(x.square() + math.log(2 * math.pi)).sum(dim=-1) / 2

def build_datasets(dataset_type:str):
    if dataset_type == 'source':
        return MNIST("./data", train=bool, transform=Compose([ToTensor(), Normalize((.5, ), (.5,))]), download=True)
    elif dataset_type == 'target':
        generator1 = torch.Generator().manual_seed(42)
        dataset =  EMNIST("./data", train=bool, split='letters', transform=Compose([ToTensor(), Normalize((.5, ), (.5,))]), download=True)
        data_train, _ = torch.utils.data.random_split(dataset, [60_000, len(dataset) - 60_000], generator=generator1)
        return data_train
    else:
        raise ValueError("Not providing dataset option")

def DDP_sampler(rank, world_size, dataset_src, dataset_dist):
    src_sampler = torch.utils.data.distributed.DistributedSampler(dataset_src,num_replicas=world_size,rank=rank) 

    dist_sampler = torch.utils.data.distributed.DistributedSampler(dataset_dist,num_replicas=world_size,rank=rank) 
    return src_sampler, dist_sampler

@torch.no_grad
def savefigs(x0, x1, flow_out):
    plt.scatter(x0.cpu()[:,0], x0.cpu()[:,1], c='g', label='source')
    plt.scatter(x1.cpu()[:,0], x1.cpu()[:,1], c='b', label='target')
    plt.scatter(flow_out.cpu()[:,0], flow_out.cpu()[:,1], c='c', label='flow_out')
    plt.legend()
    plt.savefig("flowout.png")

def init_process(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1234'
    init_process_group("nccl", rank = rank, world_size=world_size)

def main(rank, world_size):
    seed_everything()
    init_process(rank, world_size)

    epoch = 500
    batch_size= 256

    model = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(rank) 
    node = NeuralODE(model, solver='dopri5', sensitivity='adjoint', atol=1e-4, rtol=1e-4)
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

    #---------source dataset----------#
    data_src = build_datasets("source")
    data_dist = build_datasets("target")

    src_sampler, dist_sampler = DDP_sampler(rank, world_size, data_src, data_dist)
    dataloader_src = torch.utils.data.DataLoader(data_src, batch_size = batch_size, sampler = src_sampler, drop_last=True)
    dataloader_dist = torch.utils.data.DataLoader(data_dist, batch_size = batch_size, sampler = dist_sampler, drop_last=True)

    model.train()
    for epoch in tqdm(range(epoch)): 
        loss_value = 0
        for iters, (src_batch, dist_batch) in tqdm(enumerate(zip(dataloader_src, dataloader_dist))):
            optimizer.zero_grad()
            x0 = src_batch[0].to(rank)
            x1 = dist_batch[0].to(rank)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt)
            loss_v = torch.mean((vt - ut)**2)
            loss_v.backward()
            loss_value += loss_v.item()
            optimizer.step()
        loss_value /= len(dataloader_src)
        with torch.no_grad():
            print(f"{loss_value=:.4f}")


    # -------- sampling ----------#

    if rank==0:
        with torch.no_grad():
            traj = node.trajectory(x0, t_span = torch.linspace(0,1,2).to(rank))
            grid_source = make_grid(x0[:100].cpu().view(-1,1,28,28).clip(-1,1), value_range=(-1,1), padding=0, nrow=10)
            grid_dist = make_grid(x1[:100].cpu().view(-1,1,28,28).clip(-1,1), value_range=(-1,1), padding=0, nrow=10)
            grid = make_grid(
                traj[-1, :100].view(-1,1,28,28).clip(-1,1), value_range=(-1,1), padding=0, nrow=10
            )

            x0 = ToPILImage()(grid_source)
            x1 = ToPILImage()(grid_dist)
            x1_infer = ToPILImage()(grid)
            #x1_infer = to_pil_image(x1[0].view(1,28,28), mode='L')
            #x0_infer.save("x0_infer.png")
            x1_infer.save("x1_infer.png")
            x0.save("x0.png")
            x1.save("x1.png")
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

