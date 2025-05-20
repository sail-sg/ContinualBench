"""Use this script to control the env with your keyboard.

For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""

import copy
import pprint
import sys
from time import sleep

import numpy as np
import pygame
import torch
import tree
import tyro
from pygame.locals import KEYDOWN, QUIT

from continual_bench.envs import ContinualBenchEnv, reward_fns


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


def main():
    pygame.init()
    pygame.display.set_mode((400, 300))

    char_to_action = {
        "w": np.array([0, -1, 0, 0]),
        "a": np.array([1, 0, 0, 0]),
        "s": np.array([0, 1, 0, 0]),
        "d": np.array([-1, 0, 0, 0]),
        "q": np.array([1, -1, 0, 0]),
        "e": np.array([-1, -1, 0, 0]),
        "z": np.array([1, 1, 0, 0]),
        "c": np.array([-1, 1, 0, 0]),
        "k": np.array([0, 0, 1, 0]),
        "j": np.array([0, 0, -1, 0]),
        "h": "close",
        "l": "open",
        "x": "toggle",
        "r": "reset",
        "0": "set_task",
        "1": "set_task",
        "2": "set_task",
        "3": "set_task",
        "4": "set_task",
        "5": "set_task",
        "t": "get_env_state",
        "y": "set_env_state",
    }

    env = ContinualBenchEnv(render_mode="human")
    env.camera_name = "corner2"
    task_name = env.task_spec.name

    lock_action = False
    random_action = False
    obs = env.reset()
    model_state = env.get_init_state()
    model_state = tree.map_structure(
        lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x, model_state
    )
    last_action = action = np.zeros(4)
    reward = -1
    infos = {}

    # f = open("tmp_gripper.csv", "a")
    while True:
        if not lock_action:
            action[:3] = 0
        if not random_action:
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit()
                if event.type == KEYDOWN:
                    char = event.dict["key"]
                    new_action = char_to_action.get(chr(char), None)
                    if isinstance(new_action, np.ndarray):
                        action[:3] = new_action[:3]
                    elif new_action == "toggle":
                        lock_action = not lock_action
                    elif new_action == "reset":
                        obs = env.reset()
                    elif new_action == "set_task":
                        try:
                            task_name = env.all_tasks[int(chr(char))]
                        except IndexError:
                            print("task out of range")
                            continue
                        env.set_task(task_name)
                        obs = env.reset()
                        model_state = env.get_init_state()
                        model_state = tree.map_structure(
                            lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x, model_state
                        )
                    elif new_action == "get_env_state":
                        env_state = get_env_state(env)
                    elif new_action == "set_env_state":
                        set_env_state(env, env_state)
                    elif new_action == "close":
                        action[3] = 1
                        # finger_right, finger_left = (
                        #     env.data.body("rightclaw").xpos,
                        #     env.data.body("leftclaw").xpos,
                        # )

                        # finger_center_y = (finger_right + finger_left) / 2
                        # # right_diff_y = finger_center - finger_right
                        # # left_diff_y = finger_left - finger_center

                        # import mujoco
                        # env.model.body_pos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "rightclaw")] = finger_right + np.array([0, 0.01, 0])
                        # env.model.body_pos[mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "leftclaw")] = finger_left - np.array([0, 0.01, 0])

                    elif new_action == "open":
                        action[3] = -1
                    else:
                        action = np.zeros(3)
                    print(action)
                    print("ob", obs)
                    print("reward", reward)
                    finger_right, finger_left = (
                        env.data.body("rightclaw").xpos,
                        env.data.body("leftclaw").xpos,
                    )
                    finger_center = (finger_right + finger_left) / 2
                    right_diff_y = finger_right - finger_center
                    left_diff_y = finger_center - finger_left

                    print("claw", right_diff_y[1], left_diff_y[1])
                    try:
                        pprint.pprint(infos[task_name])
                    except KeyError:
                        pass
                    try:
                        reward_fn = getattr(reward_fns, f"{task_name}_reward_fn")
                        model_state["obs"] = torch.from_numpy(obs[None, ...]).float()
                        hf_reward = reward_fn(torch.from_numpy(last_action[None, ...]).float(), model_state)
                        diff = hf_reward - reward[task_name]
                        print("reward diff", diff)
                    except Exception:
                        print("cannot find reward fn")
        else:
            action = env.action_space.sample()
        last_action = action
        obs, reward, trunc, infos = env.step(action)
        # openness = obs[3]
        # left_pad = env.get_body_com("leftpad")
        # right_pad = env.get_body_com("rightpad")
        # y_tcp_center = env.tcp_center[1]
        # y_hand_center = obs[1]
        # f.write(f"{openness},{y_tcp_center},{y_hand_center},{left_pad[1]},{right_pad[1]}\n")
        env.curr_path_length = 0
        sleep(0.1)
        env.render()


if __name__ == "__main__":
    tyro.cli(main)
