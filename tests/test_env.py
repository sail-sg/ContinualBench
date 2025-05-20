import time

import numpy as np
import torch
import tree
import tyro
from PIL import Image

from continual_bench.envs import ContinualBenchEnv, reward_fns


def run(test_reward_fn: str = "", task: str = "door", seed: int = 42, render: bool = False, human: bool = False):
    env = ContinualBenchEnv(render_mode="human" if human else "rgb_array")

    env.set_task(task)
    env.reset()
    model_state = tree.map_structure(
        lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x, env.get_init_state()
    )
    frames = []
    print(model_state)
    for _ in range(2):
        action = np.zeros(4)
        # action = env.action_space.sample()
        action[3] = np.random.choice([-1, 1, 0])
        next_obs, reward_dict, _, _ = env.step(action)
        print(next_obs)
        print("qpos:", env.data.qpos.flat.copy())
        # debug_fun(env_name, env, action, next_obs)
        if test_reward_fn:
            reward = reward_dict[test_reward_fn]
            reward_fn = getattr(reward_fns, f"{test_reward_fn}_reward_fn")
            model_state["obs"] = torch.from_numpy(next_obs[None, ...]).float()
            hf_reward = reward_fn(torch.from_numpy(action[None, ...]).float(), model_state)
            print(reward, hf_reward, hf_reward - reward)
        if render and not human:
            frames.append(env.render())
    if human:
        while True:
            time.sleep(0.03)
            env.render()

    if frames:
        imgs = [Image.fromarray(f) for f in frames]
        save_fn = "output/test_env.gif"
        print("save to", save_fn)
        imgs[0].save(
            save_fn,
            save_all=True,
            append_images=imgs[1:],
            duration=10,
            loop=0,
        )


if __name__ == "__main__":
    tyro.cli(run)
