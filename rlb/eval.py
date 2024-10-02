import os, sys
import numpy as np
import os.path as osp
import logging
import csv
import shlex
from multiprocessing import Value
from utils import DictConfig, configurable
from utils.env import rlbench_obs_config, EndEffectorPoseViaPlanning, CustomMultiTaskRLBenchEnv
from utils.metric import StatAccumulator
from utils.structure import RLBENCH_TASKS
from utils.rollout import RolloutGenerator
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper


@configurable()
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("bash> " + " ".join(map(shlex.quote, sys.argv)))
    log_dir = cfg.output_dir
    logger.info(f"Log dir: {log_dir}") 
    os.makedirs(log_dir, exist_ok=True)
    env_cfg = cfg.env
    obs_config = rlbench_obs_config(env_cfg.cameras, [env_cfg.image_size, env_cfg.image_size], method_name="")
    
    device = cfg.eval.device

    py_module = cfg.py_module
    from importlib import import_module
    MOD = import_module(py_module)
    Policy, PolicyNetwork = MOD.Policy, MOD.PolicyNetwork

    net = PolicyNetwork(cfg.model.hp, cfg.env, render_device=f"cuda:{device}").to(device)
    agent = Policy(net, cfg.model.hp, log_dir=log_dir)

    agent.build(training=False, device=device)
    agent.load(cfg.model.weights)
    agent.eval()
    if hasattr(agent, 'load_clip'):
        agent.load_clip()

    gripper_mode = Discrete()
    arm_action_mode = EndEffectorPoseViaPlanning()
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [t.replace(".py", "") for t in os.listdir(rlbench_task.TASKS_PATH)
                if t != "__init__.py" and t.endswith(".py")]
    
    
    if env_cfg.tasks == 'all':
        tasks = RLBENCH_TASKS
    else:
        tasks = env_cfg.tasks.split(',')

    task_classes = []
    for task in tasks:
        if task not in task_files:
            raise ValueError("Task %s not recognised!." % task)
        task_classes.append(task_file_to_task_class(task))
    

    eval_env = CustomMultiTaskRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=cfg.eval.datafolder,
        episode_length=cfg.env.episode_length,
        headless=cfg.eval.headless,
        swap_task_every=cfg.eval.episode_num,
        include_lang_goal_in_obs=True,
        time_in_state=cfg.env.time_in_state,
        record_every_n= 1 if cfg.eval.save_video else -1,
        origin_style_state=cfg.env.origin_style_state
    )
    eval_env.eval = True

    csv_file = "eval_results.csv"
    if not osp.exists(osp.join(log_dir, csv_file)):
        with open(osp.join(log_dir, csv_file), "w") as csv_fp:
            fieldnames = ["task", "success rate", "length", "total_transitions"]
            csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
            csv_writer.writeheader()

    stats_accumulator = StatAccumulator()
    eval_env.launch()

    current_task_id = -1
    num_tasks = len(tasks)
    step_signal = Value("i", -1)
    scores = {}
    rollout = RolloutGenerator(device=cfg.eval.device)

    
    for task_id in range(num_tasks):
        task_rewards = []
        for ep in range(cfg.eval.episode_num):
            episode_rollout = []
            generator = rollout.generator(
                step_signal=step_signal,
                env=eval_env,
                agent=agent,
                episode_length=cfg.env.episode_length,
                eval=True,
                eval_seed=ep,
                record_enabled=False
            )
            try:
                for transition in generator:
                    episode_rollout.append(transition)
            except StopIteration as e:
                continue
            except Exception as e:
                eval_env.shutdown()
                raise e

            for transition in episode_rollout:
                stats_accumulator.step(transition, True)
                current_task_id = transition.info["active_task_id"]
                assert current_task_id == task_id

            task_name = tasks[task_id]
            reward = episode_rollout[-1].reward
            task_rewards.append(reward)
            lang_goal = eval_env._lang_goal
            logger.info(
                f"Evaluating {task_name} | Episode {ep} | Score: {reward} | Episode Length: {len(episode_rollout)} | Lang Goal: {lang_goal}"
            )

        # report summaries
        summaries = []
        summaries.extend(stats_accumulator.pop())
        task_name = tasks[task_id]

        # writer csv first
        with open(os.path.join(log_dir, csv_file), "a") as csv_fp:
            fieldnames = ["task", "success rate", "length", "total_transitions"]
            csv_writer = csv.DictWriter(csv_fp, fieldnames=fieldnames)
            csv_results = {"task": task_name}
            for s in summaries:
                if s.name == "eval_envs/return":
                    csv_results["success rate"] = s.value
                elif s.name == "eval_envs/length":
                    csv_results["length"] = s.value
                elif s.name == "eval_envs/total_transitions":
                    csv_results["total_transitions"] = s.value
                if "eval" in s.name:
                    s.name = "%s/%s" % (s.name, task_name)
            csv_writer.writerow(csv_results)
        
        if len(summaries) > 0:
            task_score = [
                s.value for s in summaries if f"eval_envs/return/{task_name}" in s.name
            ][0]
        else:
            task_score = "unknown"
        scores[task_name] = task_score

    eval_env.shutdown()
    csv_fp.close()
    logger.info(scores)
    logger.info(f'average success rate: {np.mean(list(scores.values()))}')


if __name__ == "__main__":
    main()