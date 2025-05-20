import copy
import multiprocessing as mp
import pprint
import time
from typing import List, Sequence

import gym
import numpy as np
import omegaconf
import torch
import tree
import tyro
from mbrl.planning import TrajectoryOptimizer
from PIL import Image
from tqdm import trange

from continual_bench.envs import ContinualBenchEnv, reward_fns

env__: gym.Env


def init(seed: int):
    global env__
    env__ = ContinualBenchEnv(seed=seed, render_mode="rgb_array")
    env__.reset()


def get_env_state(env):
    joint_state = (env.data.qpos, env.data.qvel)
    mocap_state = (env.data.mocap_pos, env.data.mocap_quat)
    state = (joint_state, mocap_state)
    return copy.deepcopy(state)


def set_env_state(env, state):
    joint_state, mocap_state = state
    qpos, qvel = joint_state
    mocap_pos, mocap_quat = mocap_state
    env.data.mocap_pos = mocap_pos
    env.data.mocap_quat = mocap_quat
    env.set_state(qpos, qvel)


def evaluate_all_action_sequences(
    task_id: str,
    action_sequences: Sequence[Sequence[np.ndarray]],
    pool: mp.Pool,
    current_state,
) -> torch.Tensor:
    res_objs = [
        pool.apply_async(evaluate_sequence_fn, (task_id, sequence, current_state))  # type: ignore
        for sequence in action_sequences
    ]
    res = [res_obj.get() for res_obj in res_objs]
    return torch.tensor(res, dtype=torch.float32)


def evaluate_sequence_fn(task_id: str, action_sequence: np.ndarray, current_state) -> float:
    global env__
    set_env_state(env__, current_state)
    rewards = []
    for i in range(len(action_sequence)):
        action = action_sequence[i]
        next_obs, reward_dict, _, _ = env__.step(action)
        reward = reward_dict[task_id]
        env__.curr_path_length = 0
        rewards.append(reward)
    return np.sum(rewards).item()


def main(
    num_processes: int = 16,
    population_size: int = 150,
    control_horizon: int = 15,
    max_steps: int = 150,
    num_iter: int = 3,
    render: bool = True,
    random: bool = False,
    method: str = "icem",
    seed: int = 0,
    task_sequence: List[str] = ["door", "window", "button"],
    verbose: bool = False,
):
    mp.set_start_method("spawn")
    env = ContinualBenchEnv(seed=seed, render_mode="rgb_array")

    if method == "cem":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.CEMOptimizer",
                "device": "cpu",
                "num_iterations": num_iter,
                "elite_ratio": 0.01,
                "population_size": population_size,
                "alpha": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
            }
        )
    elif method == "mppi":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.MPPIOptimizer",
                "num_iterations": num_iter,
                "gamma": 1.0,
                "population_size": population_size,
                "sigma": 0.95,
                "beta": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
                "device": "cpu",
            }
        )
    elif method == "icem":
        optimizer_cfg = omegaconf.OmegaConf.create(
            {
                "_target_": "mbrl.planning.ICEMOptimizer",
                "num_iterations": num_iter,
                "elite_ratio": 0.1,
                "population_size": population_size,
                "population_decay_factor": 1.25,
                "colored_noise_exponent": 2.0,
                "keep_elite_frac": 0.1,
                "alpha": 0.1,
                "lower_bound": "???",
                "upper_bound": "???",
                "return_mean_elites": "true",
                "device": "cpu",
            }
        )
    else:
        raise ValueError
    controller = TrajectoryOptimizer(
        optimizer_cfg,
        env.action_space.low,
        env.action_space.high,
        control_horizon,
        # keep_last_solution=False,
    )

    task_idx = 0

    all_tasks = task_sequence
    env.set_task(all_tasks[task_idx])
    env.reset()  # Reset environment
    model_state = tree.map_structure(
        lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x, env.get_init_state()
    )
    obs_records = []
    with mp.Pool(processes=num_processes, initializer=init, initargs=[seed]) as pool:
        total_reward = 0
        frames = []
        for _t in trange(max_steps):
            curr_task = all_tasks[task_idx]

            if render:
                frames.append(env.render())
                # frames.append(env.render())
            curr_state = get_env_state(env)

            def obj_fun(action_sequences: torch.Tensor):
                return evaluate_all_action_sequences(
                    curr_task,
                    action_sequences.numpy(),
                    pool,
                    curr_state,
                )

            if random:
                action = env.action_space.sample()
            else:
                plan = controller.optimize(obj_fun)
                action = plan[0]
            next_obs, reward_dict, trunc, info = env.step(action)
            obs_records.append(next_obs)
            reward = reward_dict[curr_task]
            total_reward += reward
            if verbose:
                print()
                print(curr_task, "action=", action, "|", total_reward, trunc)
                print(next_obs)
                pprint.pprint(info[curr_task])

            # Check reward function
            reward_fn = getattr(reward_fns, f"{curr_task}_reward_fn")

            model_state["obs"] = torch.from_numpy(next_obs[None, ...]).float()
            hf_reward = reward_fn(torch.from_numpy(action[None, ...]).float(), model_state)
            diff = hf_reward - reward
            print("reward diff", diff)
            warning_color = "\033[93m"
            end_c = "\033[0m"
            if torch.abs(diff) > 0.1:
                print(f"{warning_color} Large reward diff! {end_c}")

            if info[curr_task]["success"]:
                if task_idx == len(all_tasks) - 1:
                    print("finish break")
                    break
                else:
                    task_idx += 1
                    controller.reset()
                    env.set_task(all_tasks[task_idx])
                    env.reset()
                    model_state = tree.map_structure(
                        lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x, env.get_init_state()
                    )
                    total_reward = 0
                    print("\nswitching to a new task", env.all_tasks[task_idx], "\n")
            if trunc:
                print("truncation break")
                break

    obs_records = np.stack(obs_records, axis=0)
    print("obs max", obs_records.max(0))
    print("obs min", obs_records.min(0))
    print("obs mean", obs_records.mean(0))
    print("obs std", obs_records.std(0))
    if frames:
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            f"output/control_continual_bench_{method}_{int(time.time())}.gif",
            save_all=True,
            append_images=imgs[1:],
            duration=10,
            loop=0,
        )


if __name__ == "__main__":
    tyro.cli(main)
