# Environment Setup

## Step-1: Conda Environment

Recreate the Python 3.8 conda environment:

```bash
conda env create -f environment.yaml 
```


### Use a Different Python Environment (Optional)

If you want to use a different python environment, for example a newer version of python. First downgrade pip, setuptools and wheel (for compatibility with `gym==0.21.0`):

```bash
pip install pip==21 setuptools==65.5.0 wheel==0.38.0
```

Then use pip to install packages from the [environment.yaml](environment.yaml), according to [pip-installing-environment-yml-as-if-its-a-requirements-txt](https://stackoverflow.com/questions/72824468/pip-installing-environment-yml-as-if-its-a-requirements-txt). 

## Step-2: Install Cuda 12

Optional. Skip if you already have cuda 12 installed and `CUDA_HOME` environment variable set. 

Download `cuda_12.3.2_545.23.08_linux.run` from [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) and install it:

```bash
bash ./cuda_12.3.2_545.23.08_linux.run --silent   --toolkit --toolkitpath=$HOME/cuda-12.3
export CUDA_HOME=$HOME/cuda-12.3 
```



## Step-3: Install Dependencies

Some packages cannot be installed from the above `conda env create` directly, we need to install them separately. 


### ALOHA GYM

```bash
pip install gym-aloha==0.1.1 --ignore-requires-python
```


### Pytorch3D

```bash
export NVCC_FLAGS="--generate-code arch=compute_80,code=sm_80 --generate-code arch=compute_86,code=sm_86 --generate-code arch=compute_87,code=sm_87 --generate-code arch=compute_89,code=sm_89" # adjust this according to your GPU

pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
```

### xformers


```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.7;8.9"  # adjust this according to your GPU
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```


To speed up the installation, you can install `ninja` and set `MAX_JOBS`: 

```bash
pip install ninja
export MAX_JOBS=1 # make it larger will speed up installation, buy may cause failure on xformers installation.
```


### RLBench 

#### CoppeliaSim & PyRep & RLBench

Please follow the instruction in https://github.com/stepjam/RLBench. 

If your server does not provide GUI and sudo permission, like mine, please follow the headless rendering tips for rlbench in [Eval.md](Eval.md).

#### Faster Point-Renderer (optional but recommended)

Install by downloading point-renderer from [RVT](https://github.com/nvlabs/rvt). I did not include it in the repo for LICENSE reason.

```bash
cd rlb
git clone https://github.com/NVlabs/RVT
cp -rf RVT/rvt/libs/point-renderer ./
rm -rf ./RVT/
cd ./point-renderer
pip install -e .
cd ../../
```


This is optional if you use pytorch3d renderer. To skip this, just set `render_with_cpp=False` in rlbench configuration files.  

But the pretrained models are trained with this renderer, setting `render_with_cpp=False` will cause a little performance drop (they render slightly different pictures). But I believe it won't affect anything if you train from scratch.
