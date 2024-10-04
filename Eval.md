Please first setup your dataset and pretrained weights. 

# Push-T

Check out [pusht/demo.ipynb](pusht/demo.ipynb). It loads and tests the pretrained model, and save the videos into `pusht/outputs/demo` folder. The full evaluation is done during training periodically.


# ALOHA

Check out [aloha/demo.ipynb](aloha/demo.ipynb). It loads and tests the pretrained model, and save the videos into `aloha/outputs/demo` folder. Like Push-T, the full evaluation is done during training periodically.



# RLBench

The following command will load ARP+ model and evaluate it on RLBench.

```bash
cd rlb
python3 eval.py config=./configs/arp_plus.yaml  model.weights=./weights/arp_plus_model_70000.pth  hydra.job.name=eval.arp_plus  eval.device=0  output_dir=outputs/eval.arp_plus/`date +"%Y-%m-%d_%H-%M"` 
```

Adding `eval.save_video=True` will save the rotating videos like the one in [assets/demo.mp4](assets/demo.mp4) to `rlb/outputs/recording`. But the evaluation will get extremely slow.

Ensure you are running `eval.py` in a machine with GUI and `DISPLAY` environment variable is set. In other cases, check the tip below.  


## Tip: Running RLBench on a Headless Server without Sudo 

The evaluation of RLBench requires a GUI environment, and `DISPLAY` environment variable set. 

If you only have access to a remote server, you can use `xvfb` to create a cpu-based virtual display. It's a little bit slow but tolerable. In case that you do not have sudo permission to install softwares. Install it with anaconda, follow scripts below: 

```bash
conda install anaconda::xorg-x11-server-xvfb-cos6-x86_64
```

The above command will install a xvfb that tailored for centos 6, but it works for other linux as well. Run xvfb with this command

```bash
$HOME/anaconda/x86_64-conda-linux-gnu/sysroot/usr/bin/Xvfb :99 -screen 0 1024x768x24 +extension GLX +render -noreset
```

`$HOME/anaconda` is where my anaconda is installed. If you have it else where, change it accordingly. Then, set `DISPLAY` environment variable:

```bash
export DISPLAY=:99
```

for the shell where you run the evaluation script.


If the `xvfb` command complains about some missing shared libraries, for example `libcrypto.so.10`, you can download it from the internet (or copy it from another machine that has this) and put it in your home directory. Then run xvfb with `LD_PRELOAD`:

```bash
LD_PRELOAD=$HOME/libcrypto.so.10  $HOME/anaconda3/x86_64-conda-linux-gnu/sysroot/usr/bin/Xvfb :99 -screen 0 1024x768x24 +extension GLX +render -noreset
```

If the above does not work for you because of some GLIB version issues (the real reason is your server being too out-dated), the last but sure resort is to use [singularity](https://github.com/sylabs/singularity) container and run evaluation inside. I have built a working one, let me know if you need it.