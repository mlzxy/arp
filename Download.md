# Dataset Download

All datasets and pretrained weights are stored in a box folder at https://rutgers.box.com/s/uzozemx67kje58ycy3lyzf1zgddz8tyq. These files are located in `datasets` and `weights` folders.

> It takes a lot of time to download each file from browser manually. Therefore, I prepare a single compressed file `all_datasets_and_weights.tar.gz` that includes all files (in that box folder). You can choose to just download that and extract it. It will extract to a folder called `release` and you can find all files inside. 


## Push-T

Directly download from the origin source. 

```bash
cd pusht
mkdir data && cd data
wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
unzip pusht.zip && rm -f pusht.zip && cd ..
```



## ALOHA

```bash
cd aloha
mkdir data && cd data

# Download datasets/aloha_human_demo_with_waypoints.zip
# from the box folder, and put it in this data folder

unzip aloha_human_demo_with_waypoints.zip 
mv aloha_human_demo_with_waypoints/lerobot .
rm -f aloha_human_demo_with_waypoints.zip aloha_human_demo_with_waypoints && cd ..
```


This dataset shares the same episodes with the original one but adds 2d waypoints. You can use the script [aloha/compute_waypoints.py](aloha/compute_waypoints.py) to re-generate this dataset from origin lerobot data.



## RLBench


```bash
cd rlb
mkdir data && cd data

# Download datasets/RLBench.tar
# from the box folder, and put it in this data folder

tar xvf RLBench.tar
rm -f RLBench.tar && cd ..
```

This dataset contains the original pre-generated RLBench train / test demonstration from [peract](https://github.com/peract/peract). However, it is much smaller (only 6GB vs hundreds of GBs). Therefore, it is much easier to get started with.

The reason is that I only keep the key frames from the original dataset. The back-story is: 

1. RLBench has a key frame extraction procedure, see `keypoint_discovery` function in [rlb/dataset.py](rlb/dataset.py). Many existing works use this code snippet. 
2. There has been a long-standing "bug", regarding data sampling, in existing works. This "bug" significantly increases the sampling ratio on key frames. Read more into this issue here: https://github.com/peract/peract/issues/6#issuecomment-1355555980.  
3. Based on my personal experience, I found only key-frames contribute to the learning of the policy. Therefore, I simplify the implementation ("fix" this "bug") and trim the training set. 
4. I optimize the code a little bit so evaluation do not read the full testing episodes, in doing so, the test set is also trimmed. 


# Pretrained Weights

Here only the instructions on downloading the models of our main results are provided. The training logs of other experiments are stored in the `training-logs` folder (in box). Those experiments are detailed in the [Experiments.md](Experiments.md)



## Push-T

```bash
cd pusht
mkdir weights

# Download weights/pusht/epoch=2000-test_mean_score=0.865.ckpt from Box
# and put it in this weights folder

cd ..
```

There are better ones, but I forget to save them... 


## ALOHA

```bash
cd aloha
mkdir weights

# download models/aloha/model.transfer_cube.safetensor and models/aloha/model.insertion.safetensors
# from the box folder, and put them in this weights folder

cd ..
```



## RLBench


```bash
cd rlb
mkdir weights

# download  models/rlb/arp_model_80000.pth and models/rlb/arp_plus_model_70000.pth
# from the box folder, and put them in this weights folder

cd ..
```