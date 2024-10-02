# Autoregressive Action Sequence Learning for Robotic Manipulation

## Environment Setup

```bash
mamba env create -f environment.yaml 
conda activate arp

bash ./cuda_12.3.2_545.23.08_linux.run --silent   --toolkit --toolkitpath=$HOME/cuda-12.3
export CUDA_HOME=$HOME/cuda-12.3 
pip install ninja
export NVCC_FLAGS="--generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_86,code=sm_86 --generate-code arch=compute_87,code=sm_87 --generate-code arch=compute_89,code=sm_89"
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9" 
export MAX_JOBS=1

pip install gym-aloha==0.1.1 --ignore-requires-python
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```


## Push-T

### Setup Dataset

```bash
cd pusht
mkdir data && cd data
wget http://45.63.16.248/datasets/pusht.zip
unzip pusht.zip && rm -f pusht.zip && cd ..
```

### Demo With Pretrained Model

Download pretrained models: 

```bash
cd pusht
mkdir weights
wget http://45.63.16.248/models/pusht/epoch%3D2000-test_mean_score%3D0.865.ckpt 
cd ..
```

Then go to [pusht/demo.ipynb](pusht/demo.ipynb), it loads and tests the pretrained model, and save the videos into `pusht/outputs/demo` folder. 

### Start Training

```bash
# for server rendering MUJOCO
export PYOPENGL_PLATFORM=egl 
export MUJOCO_GL=egl

# feel free to change it to your device id
export CUDA_VISIBLE_DEVICES=0
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0
timestamp=`date +"%y-%m-%d_%H_%M_%S"`

cd ./pusht
python3 ./train.py --config-dir ./configs  --config-name arp.yaml  hydra.run.dir=outputs/arp/${timestamp} \
    training.device=cuda:0  logging.mode=offline  logging.name="arp@${timestamp}"  name=arp
```

## ALOHA

### Setup Dataset 

```bash
cd aloha
mkdir data && cd data
wget http://45.63.16.248/datasets/aloha_human_demo_with_waypoints.zip
unzip aloha_human_demo_with_waypoints.zip 
mv aloha_human_demo_with_waypoints/lerobot .
rm -f aloha_human_demo_with_waypoints.zip aloha_human_demo_with_waypoints && cd ..
```

### Demo With Pretrained Model

Download pretrained model:

```bash
cd aloha
mkdir weights
wget http://45.63.16.248/models/aloha/model.transfer_cube.safetensors
wget http://45.63.16.248/models/aloha/model.insertion.safetensors
cd ..
```

Then go to [aloha/demo.ipynb](aloha/demo.ipynb), it loads and tests the pretrained model, and save the videos into `aloha/outputs/demo` folder. 

### Start Training

```bash
# for server rendering MUJOCO
export PYOPENGL_PLATFORM=egl 
export MUJOCO_GL=egl

# feel free to change it to your device id
export CUDA_VISIBLE_DEVICES=0
export EGL_DEVICE_ID=0
export MUJOCO_EGL_DEVICE_ID=0
timestamp=`date +"%y-%m-%d_%H_%M_%S"`

export task=insertion # or set to `transfer_cube`

if [[ "$task" == "insertion" ]]; then
    export aloha_env="AlohaInsertion"
elif [[ "$task" == "transfer_cube" ]]; then
    export aloha_env="AlohaTransferCube"
else
    echo "unrecognized task!"
fi

cd ./aloha
run_dir=outputs/${task}-arp/${timestamp}
mkdir -p ${run_dir}

python3 train.py --config-dir ./configs \
    --config-name arp  device=cuda:0  \
    hydra.run.dir="${run_dir}"   hydra.job.name="${task}-arp@${timestamp}" \
    env.task=${aloha_env}-v0  dataset_repo_id=lerobot/aloha_sim_${task}_human seed=$RANDOM
```


## RLBench

### Setup Dataset

```bash
cd rlb
mkdir data && cd data
wget http://45.63.16.248/datasets/RLBench.tar # 6GB
tar xvf RLBench.tar
rm -f RLBench.tar && cd ..
```


### Evaluation With Pretrained Model

Download pretrained model:

```bash
cd rlb
mkdir weights
wget http://45.63.16.248/models/rlb/arp_model_80000.pth
wget http://45.63.16.248/models/rlb/arp_plus_model_70000.pth
cd ..
```

Ensure you are running `eval.py` in a machine with a X-Server and `DISPLAY` environment variable is set. 

```bash
python3 eval.py config=./configs/arp_plus.yaml  model.weights=./weights/arp_plus_model_70000.pth  hydra.job.name=eval.arp_plus  eval.device=0  output_dir=outputs/eval.arp_plus/`date +"%Y-%m-%d_%H-%M"` 
```


### Training

```bash
# by default using 2 gpus and batch size on each gpu is 96
num_gpus=${num_gpus:-2}
bs=${bs:-96} 

python3 train.py config=./configs/arp_plus.yaml  hydra.job.name=arp_plus  train.num_gpus=${num_gpus}  train.bs=${bs} 
```

