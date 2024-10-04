import wandb
import os
import os.path as osp
import torch
from copy import copy
from tqdm import tqdm
import logging
from time import time
import sys, shlex
from utils import configurable, DictConfig, config_to_dict
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from utils.object import to_device
from utils.stat_dict import StatisticsDict
from dataset import RealRobotDataset
from network import RobotPolicy

@configurable()
def main(cfg: DictConfig):
    log_dir = cfg.output_dir
    if cfg.wandb:
        wandb.init(project=cfg.wandb, name='/'.join(log_dir.split('/')[-2:]), config=config_to_dict(cfg))
    logfile = open(osp.join(log_dir, 'log.txt'), "w")

    device = cfg.device
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    def logger(msg, printer=print):
        print(msg, file=logfile, flush=True)
        printer(msg)
    
    dataset = RealRobotDataset(cfg.data.folder, cfg.data.train_episodes, offset_z=0.0, batch_num=cfg.train.batch_size * cfg.train.num_steps)
    eval_dataset = RealRobotDataset(cfg.data.folder, cfg.data.eval_episodes, offset_z=0.0)

    cfg.train.lr *= cfg.train.batch_size

    agent = RobotPolicy(cfg.model, device)
    agent.build(cfg.train)
    agent.train()
    agent.to(device)

    dataloader = dataset.dataloader(cfg.train.batch_size, num_workers=cfg.train.num_workers)
    eval_dataloader = eval_dataset.dataloader(cfg.train.batch_size, num_workers=0, shuffle=False)

    train_stats, eval_stats = StatisticsDict(), StatisticsDict()
    steps = 0
    for batch in tqdm(dataloader):
        batch = {k:to_device(v, device) for k,v in batch.items()}
        loss_dict = agent.forward_train(batch, backprop=True)

        train_stats.push(loss_dict)
        if cfg.wandb : wandb.log(train_stats.current)

        if steps % cfg.train.disp_freq == 0:
            running_stat = train_stats.running
            running_stat['lr'] = agent.learning['optimizer'].param_groups[0]['lr']
            logger(f"[step:{str(steps).zfill(8)} time:{time():.01f}s] " + " ".join([f"{k}:{v:.04f}" for k, v in sorted(running_stat.items())]),
                printer=tqdm.write)
        if steps != 0 and steps % cfg.train.save_freq == 0:
            logger(f"checkpoint to {log_dir} at step {steps} and reset running metrics", printer=tqdm.write)
            agent.save(steps, log_dir)
            train_stats.reset()
        
        if steps % cfg.train.eval_freq == 0:
            agent.eval()
            with torch.no_grad():
                for eval_batch in tqdm(eval_dataloader):
                    eval_batch = {k: to_device(v, device) for k,v in eval_batch.items()}
                    eval_loss_dict = agent.forward_train(eval_batch, backprop=False)
                    eval_stats.push(eval_loss_dict)
                    
                logger(f"[********* eval **********] " + " ".join([f"{k}:{v:.04f}" for k, v in sorted(eval_stats.running.items())]), printer=tqdm.write)
                eval_stats.reset()
            agent.train()
            
        steps += 1
    agent.save(steps, log_dir)   
        

if __name__ == "__main__":
    main()