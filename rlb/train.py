# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

import os
import wandb
import os.path as osp
import torch
from copy import copy
from tqdm import tqdm
import logging
from time import time
import sys, shlex
from utils import configurable, DictConfig, config_to_dict
import torch.multiprocessing as mp
from utils.dist import find_free_port
import torch.distributed as dist
from dataset import TransitionDataset
from utils.structure import RLBENCH_TASKS
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from runstats import Statistics


def train(agent, dataloader: DataLoader, logger, device: int, freq: int = 30, rank: int = 0, save_freq: int = 6000, start_step=0, use_wandb=False):
    start = time()
    run_stats = {}
    steps = start_step
    for i, batch in enumerate(tqdm(dataloader, disable=rank != 0)):
        batch = {k:v.to(device) for k,v in batch.items()}
        loss_dict = agent.update(batch)

        for k, v in loss_dict.items():
            if 'loss' in k and k not in run_stats:
                run_stats[k] = Statistics()
        stat_dict = copy(loss_dict)
        if use_wandb and rank == 0: wandb.log(stat_dict)
        for k in run_stats:
            run_stats[k].push(loss_dict[k])
            stat_dict[k] = run_stats[k].mean()
        if i % freq == 0 and rank == 0:
            logger(f"[step:{str(steps).zfill(8)} time:{time()-start:.01f}s] " + " ".join([f"{k}:{v:.04f}" for k, v in sorted(stat_dict.items())]),
                printer=tqdm.write)
        if rank == 0 and i != 0 and i % save_freq == 0:
            logger(f"checkpoint to {agent.log_dir} at step {steps} and reset running metrics", printer=tqdm.write)
            agent.save(steps)
            run_stats = {}
        steps += 1
    agent.save(steps)   


def main_single(rank: int, cfg: DictConfig, port: int, log_dir:str):
    if cfg.wandb and rank == 0:
        wandb.init(project=cfg.wandb, name='/'.join(log_dir.split('/')[-2:]), config=config_to_dict(cfg))

    world_size = cfg.train.num_gpus
    assert world_size > 0
    ddp, on_master = world_size > 1, rank == 0
    print(f'Rank - {rank}, master = {on_master}')
    if ddp:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = rank
    if on_master:
        logfile = open(osp.join(log_dir, 'log.txt'), "w")

    def log(msg, printer=print):
        if on_master:
            print(msg, file=logfile, flush=True)
            printer(msg)

    env_cfg = cfg.env
    if env_cfg.tasks == 'all':
        tasks = RLBENCH_TASKS
    else:
        tasks = env_cfg.tasks.split(',')

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()

    cfg.model.hp.lr *= (world_size * cfg.train.bs)
    cfg.model.hp.cos_dec_max_step = cfg.train.epochs * cfg.train.num_transitions_per_epoch // cfg.train.bs // world_size

    py_module = cfg.py_module
    from importlib import import_module
    MOD = import_module(py_module)
    Policy, PolicyNetwork = MOD.Policy, MOD.PolicyNetwork

    net = PolicyNetwork(cfg.model.hp, cfg.env, render_device=f"cuda:{device}").to(device)
    if ddp:
        net = DistributedDataParallel(net, device_ids=[device])
    agent = Policy(net, cfg.model.hp, log_dir=log_dir)
    agent.build(training=True, device=device)

    start_step = 0
    if cfg.model.weights:
        start_step = agent.load(cfg.model.weights)
        log(f"Resuming from step {start_step}")
    if ddp: dist.barrier()

    total_batch_num = cfg.train.num_transitions_per_epoch * cfg.train.epochs // cfg.train.bs #(cfg.train.bs * world_size)
    total_batch_num -= (start_step * world_size)
    dataset = TransitionDataset(cfg.train.demo_folder, tasks, cameras=env_cfg.cameras,
            batch_num=total_batch_num, batch_size=cfg.train.bs, scene_bounds=env_cfg.scene_bounds,
            voxel_size=env_cfg.voxel_size, rotation_resolution=env_cfg.rotation_resolution,
            cached_data_path=cfg.train.cached_dataset_path, time_in_state=cfg.env.time_in_state,
            episode_length=cfg.env.episode_length, k2k_sample_ratios=cfg.train.k2k_sample_ratios, 
            origin_style_state=cfg.env.origin_style_state)

    log("Begin Training...")
    dataloader, sampler = dataset.dataloader(num_workers=cfg.train.num_workers, 
                                             pin_memory=False, distributed=ddp)
    log(f"Total number of batches: {len(dataloader)}")

    if ddp: sampler.set_epoch(0)
    if cfg.train.eval_mode:
        agent.eval()
        torch.set_grad_enabled(False)
    else:
        agent.train()

    train(agent, dataloader, log, device, freq=cfg.train.disp_freq, rank=rank, save_freq=cfg.train.save_freq, 
        start_step=start_step, use_wandb=cfg.wandb and rank == 0)


@configurable()
def main(cfg: DictConfig):
    if cfg.train.num_gpus <= 1:
        main_single(0, cfg, -1, cfg.output_dir)
    else:
        port = find_free_port()
        mp.spawn(main_single, args=(cfg, port, cfg.output_dir),  nprocs=cfg.train.num_gpus, join=True)

if __name__ == "__main__":
    main()
