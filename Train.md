Please first setup your dataset and pretrained weights. Our logs are stored in `training-logs/main-results/` folder (in box). 

> Feel free to organize the following snippets in bash script.

# Push-T

Set cuda device

```bash
export CUDA_VISIBLE_DEVICES=0
```

Set environment variables for MUJOCO server rendering

```bash
export PYOPENGL_PLATFORM=egl 
export MUJOCO_GL=egl
export EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES
export MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES
```

Then starts training with: 

```bash
cd ./pusht

timestamp=`date +"%y-%m-%d_%H_%M_%S"`
python3 ./train.py --config-dir ./configs  --config-name arp.yaml  hydra.run.dir=outputs/arp/${timestamp} \
    training.device=cuda:0  logging.mode=offline  logging.name="arp@${timestamp}"  name=arp
```

If you have wandb configured, set `logging.mode=online`.



# ALOHA

Set cuda device

```bash
export CUDA_VISIBLE_DEVICES=0
```

Set environment variables for MUJOCO server rendering

```bash
export PYOPENGL_PLATFORM=egl 
export MUJOCO_GL=egl
export EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES
export MUJOCO_EGL_DEVICE_ID=$CUDA_VISIBLE_DEVICES
```

Then, select task: 

```bash
export task=insertion # or set to `transfer_cube`

if [[ $task == insertion ]]; then
    export aloha_env=AlohaInsertion
elif [[ $task == transfer_cube ]]; then
    export aloha_env=AlohaTransferCube
fi
echo "ALOHA ENV: $aloha_env"
```


Next, start training with: 

```bash
cd ./aloha
timestamp=`date +"%y-%m-%d_%H_%M_%S"`
run_dir=outputs/${task}-arp/${timestamp}
mkdir -p ${run_dir}

python3 train.py --config-dir ./configs  --config-name arp  device=cuda:0  \
    hydra.run.dir="${run_dir}"   hydra.job.name="${task}-arp@${timestamp}" \
    env.task=${aloha_env}-v0  dataset_repo_id=lerobot/aloha_sim_${task}_human seed=$RANDOM
```



# RLBench

Suppose you have 2 GPUs, and you want to have a batch size of 96 on each GPU. 

```bash
python3 train.py config=./configs/arp_plus.yaml  hydra.job.name=arp_plus  train.num_gpus=2 train.bs=96
```

If you have more GPUs, feel free to change the number of GPUs and batch size accordingly. 

