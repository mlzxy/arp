Here we document some experiments that are included in the paper. For most experiments, we only list the configuration files without repeating the training command (see [Train.md](Train.md) for detailed commands).

Our training & eval logs can be found at `training-logs` in the box folder https://rutgers.box.com/s/uzozemx67kje58ycy3lyzf1zgddz8tyq. 

# Baselines

- Push-T (Diffusion Policy [T]): `pusht/configs/experiments/image_pusht_diffusion_policy_trans.yaml`

- ALOHA (ACT): `aloha/configs/experiments/act.yaml`
    > Note the original ACT mentions 7 decoding layers but only uses 1 due to a code issue. We set the number of decoding layers to be 4. 

- RLBench (RVT): `rlb/configs/rvt2.yaml`. 

    The pretrained model of RVT2 can be found at `weights/rlb/rvt/rvt2_without_time_model_70000.pth`. You can run the evaluation with the following command (download the model first):  

    ```bash
    python3 eval.py config=./configs/rvt2.yaml  model.weights=./weights/rvt/rvt2_without_time_model_70000.pth 
    ```

    Note you can also run the official rvt/rvt2 models. These models are in the same folder `weights/rlb/rvt` (in box). Their config files are `rlb/configs/rvt1.official.yaml` and `rlb/configs/rvt2.official.yaml`. 

    ```bash
    # rvt
    python3 eval.py config=./configs/rvt1.official.yaml  model.weights=./weights/rvt/rvt1_official_model_14.pth

    # rvt2
    python3 eval.py config=./configs/rvt2.official.yaml  model.weights=./weights/rvt/rvt2_official_model_99.pth
    ```
    
    
    They shall reproduce the same results as in their papers. To change the output folder, set up headless rendering, find more detailed command in [Eval.md](eval.md).


# One-step Prediction

- Push-T: `pusht/configs/experiments/one_step_prediction.yaml`

- ALOHA: `aloha/configs/experiments/one_step_prediction.yaml`

# Chunk Sizes

- Push-T: 
    - `pusht/configs/experiments/chunk_size/high_level_plan_c{1,2,3,4}.yaml`
    - `pusht/configs/experiments/chunk_size/low_level_action_c{1,2,4,8}.yaml`


# Existing Methods on Different Environments

- Diffusion Policy on ALOHA (does not work well): `aloha/configs/experiments/diffusion_policy.yaml`

- ACT on Push-T: `pusht/configs/experiments/pusht_act.yaml`

- ACT on RLBench: `rlb/configs/act.yaml`
