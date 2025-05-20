from typing import Dict

import torch

from continual_bench.envs import torch_reward_utils as reward_utils

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _get_tcp_center(obs: torch.Tensor):
    tcp = obs[:, :3].clone()
    tcp[:, -1] -= 0.045  # Hand -> end-effector: ASSUME vertical
    return tcp


def _get_left_pad_y(obs: torch.Tensor):
    """y_left - y_tcp = 0.0498818 * openness - 0.00295677."""
    y_tcp = obs[:, 1]  # approximately
    openness = obs[:, 3]
    y_left = y_tcp + 0.0498818 * openness - 0.00295677
    return y_left


def _get_right_pad_y(obs: torch.Tensor):
    """y_right - y_tcp = -0.04990992 * openness + 0.00292741."""
    y_tcp = obs[:, 1]  # approximately
    openness = obs[:, 3]
    y_right = y_tcp - 0.04990992 * openness + 0.00292741
    return y_right


def button_reward_fn(act: torch.Tensor, next_model_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    del act
    next_obs = next_model_state["obs"]
    init_data = next_model_state["init_data"]
    init_tcp = next_model_state["init_tcp"]

    obj = next_obs[:, 4:7]
    tcp = _get_tcp_center(next_obs)
    target_pos = init_data["button"].target_pos
    obj_to_target_init = init_data["button"].obj_to_target_init

    tcp_to_obj = torch.norm(obj - tcp, dim=-1)
    tcp_to_obj_init = torch.norm(obj - init_tcp, dim=-1)
    obj_to_target = torch.abs(target_pos[:, 2] - obj[:, 2])

    tcp_closed = 1 - next_obs[:, 3]
    near_button = reward_utils.tolerance(
        tcp_to_obj,
        bounds=(0, 0.01),
        margin=tcp_to_obj_init,
        sigmoid="long_tail",
    )
    button_pressed = reward_utils.tolerance(
        obj_to_target,
        bounds=(0, 0.005),
        margin=obj_to_target_init,
        sigmoid="long_tail",
    )

    reward = 5 * reward_utils.hamacher_product(tcp_closed, near_button)
    reward = torch.where(tcp_to_obj <= 0.03, reward + 5 * button_pressed, reward)
    return reward.view(-1, 1)


_DOOR_OFFSET = torch.tensor([[-0.05, 0, 0]], device=_DEVICE)
_HAND_DOOR_OFFSET = torch.tensor([[0.05, 0.03, -0.01]], device=_DEVICE)


def door_reward_fn(act: torch.Tensor, next_model_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    next_obs = next_model_state["obs"]
    init_data = next_model_state["init_data"]

    theta = next_obs[:, 10]
    target_pos_x = init_data["door"].target_pos[:, 0]

    # 1. grab effort
    reward_grab = (torch.clip(act[:, 3], -1, 1) + 1.0) / 2.0

    # 2. reward pos
    hand = next_obs[:, :3]
    door = next_obs[:, 7:10] + _DOOR_OFFSET

    threshold = 0.12
    radius = torch.norm(hand[:, :2] - door[:, :2], dim=1)
    floor = torch.where(radius <= threshold, torch.zeros_like(radius), 0.04 * torch.log(radius - threshold) + 0.4)
    above_floor = torch.where(
        hand[:, 2] >= floor,
        torch.ones_like(radius),
        reward_utils.tolerance(
            floor - hand[:, 2],
            bounds=(0.0, 0.01),
            margin=floor / 2.0,
            sigmoid="long_tail",
        ),
    )
    in_place = reward_utils.tolerance(
        torch.norm(hand - door - _HAND_DOOR_OFFSET, dim=1),
        bounds=(0, threshold / 2.0),
        margin=0.5,
        sigmoid="long_tail",
    )
    ready_to_open = reward_utils.hamacher_product(above_floor, in_place)

    door_angle = -theta
    a = 0.2  # Relative importance of just *trying* to open the door at all
    b = 0.8  # Relative importance of fully opening the door
    opened = a * (theta < -torch.pi / 90.0).float() + b * reward_utils.tolerance(
        torch.pi / 2.0 + torch.pi / 6 - door_angle,
        bounds=(0, 0.5),
        margin=torch.pi / 3.0,
        sigmoid="long_tail",
    )

    reward = 2.0 * reward_utils.hamacher_product(ready_to_open, reward_grab) + 8.0 * opened
    reward = torch.where(torch.abs(next_obs[:, 7] - target_pos_x) <= 0.08, 10 * torch.ones_like(reward), reward)
    return reward.view(-1, 1)


def window_reward_fn(act: torch.Tensor, next_model_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    _TARGET_RADIUS = 0.05
    del act
    next_obs = next_model_state["obs"]
    init_data = next_model_state["init_data"]
    init_tcp = next_model_state["init_tcp"]

    obj = next_obs[:, 11:14]
    tcp = _get_tcp_center(next_obs)
    target = init_data["window"].target_pos
    window_handle_pos_init = init_data["window"].window_handle_pos_init

    target_to_obj = obj[:, 1] - target[:, 1]
    target_to_obj = torch.norm(target_to_obj, dim=-1)
    target_to_obj_init = window_handle_pos_init[:, 1] - target[:, 1]
    target_to_obj_init = torch.norm(target_to_obj_init, dim=-1)

    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0, _TARGET_RADIUS),
        margin=abs(target_to_obj_init - _TARGET_RADIUS),
        sigmoid="long_tail",
    )

    handle_radius = 0.02
    tcp_to_obj = torch.norm(obj - tcp, dim=-1)
    tcp_to_obj_init = torch.norm(window_handle_pos_init - init_tcp, dim=-1)
    reach = reward_utils.tolerance(
        tcp_to_obj,
        bounds=(0, handle_radius),
        margin=abs(tcp_to_obj_init - handle_radius),
        sigmoid="gaussian",
    )

    reward = 10 * reward_utils.hamacher_product(reach, in_place)
    return reward.view(-1, 1)


def faucet_reward_fn(act: torch.Tensor, next_model_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    target_radius = 0.07
    del act
    next_obs = next_model_state["obs"]
    init_data = next_model_state["init_data"]
    init_tcp = next_model_state["init_tcp"]

    obj = next_obs[:, 14:17]
    tcp = _get_tcp_center(next_obs)
    target = init_data["faucet"].target_pos
    obj_init_pos = init_data["faucet"].obj_init_pos

    target_to_obj = obj - target
    target_to_obj = torch.norm(target_to_obj, dim=-1)
    target_to_obj_init = obj_init_pos - target
    target_to_obj_init = torch.norm(target_to_obj_init, dim=-1)

    in_place = reward_utils.tolerance(
        target_to_obj,
        bounds=(0, target_radius),
        margin=abs(target_to_obj_init - target_radius),
        sigmoid="long_tail",
    )

    faucet_reach_radius = 0.01
    tcp_to_obj = torch.norm(obj - tcp, dim=-1)
    tcp_to_obj_init = torch.norm(obj_init_pos - init_tcp, dim=-1)
    reach = reward_utils.tolerance(
        tcp_to_obj,
        bounds=(0, faucet_reach_radius),
        margin=abs(tcp_to_obj_init - faucet_reach_radius),
        sigmoid="gaussian",
    )

    reward = 2 * reach + 3 * in_place
    reward *= 2
    reward = torch.where(target_to_obj <= target_radius, 10 * torch.ones_like(reward), reward)
    return reward.view(-1, 1)


def peg_reward_fn(act: torch.Tensor, next_model_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    next_obs = next_model_state["obs"]
    init_data = next_model_state["init_data"]
    init_tcp = next_model_state["init_tcp"]

    obj = next_obs[:, 17:20]
    tcp_opened = next_obs[:, 3]
    tcp = _get_tcp_center(next_obs)
    target = init_data["peg"].target_pos
    obj_init_pos = init_data["peg"].obj_init_pos

    tcp_to_obj = torch.norm(obj - tcp, dim=-1)
    obj_to_target = torch.norm(obj - target, dim=-1)
    pad_success_margin = 0.05
    object_reach_radius = 0.01
    x_z_margin = 0.005
    obj_radius = 0.025

    left_pad_y = _get_left_pad_y(next_obs)
    right_pad_y = _get_right_pad_y(next_obs)

    # print("pred left_pad", left_pad_y)
    # print("pred right_pad", right_pad_y)
    # print("pred tcp", tcp)

    # left_pad_y = next_obs[:, -5]
    # right_pad_y = next_obs[:, -4]
    # print("obs left_pad", left_pad_y)
    # print("obs right_pad", right_pad_y)

    # tcp = next_obs[:, -3:]
    # print("obs tcp", tcp)
    # print("obs init_tcp", init_tcp)
    # print("obs obj", obj)
    # print("obs obj_init_pos", obj_init_pos)

    object_grasped = reward_utils.gripper_caging_reward(
        left_pad_y,
        right_pad_y,
        act,
        tcp,
        init_tcp,
        obj,
        obj_init_pos,
        object_reach_radius=object_reach_radius,
        obj_radius=obj_radius,
        pad_success_thresh=pad_success_margin,
        xz_thresh=x_z_margin,
        desired_gripper_effort=0.8,
        high_density=True,
    )
    in_place_margin = torch.norm(obj_init_pos - target, dim=-1)

    # print("our object_grasped", object_grasped)

    in_place = reward_utils.tolerance(
        obj_to_target,
        bounds=(0, 0.05),
        margin=in_place_margin,
        sigmoid="long_tail",
    )
    grasp_success = (tcp_opened > 0.5) * (obj[:, 0] - obj_init_pos[:, 0] > 0.015)

    reward = 2 * object_grasped

    reward = torch.where(grasp_success * (tcp_to_obj < 0.035), 1 + 2 * object_grasped + 5 * in_place, reward)
    reward = torch.where(obj_to_target <= 0.05, 10 * torch.ones_like(reward), reward)
    return reward.view(-1, 1)


def block_reward_fn(act: torch.Tensor, next_model_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    _TARGET_RADIUS = 0.05
    next_obs = next_model_state["obs"]
    init_data = next_model_state["init_data"]
    init_tcp = next_model_state["init_tcp"]

    obj = next_obs[:, 20:23]
    tcp_opened = next_obs[:, 3]
    tcp = _get_tcp_center(next_obs)
    # target = init_data["block"].target_pos
    obj_init_pos = init_data["block"].obj_init_pos
    midpoint = init_data["block"].target_pos

    tcp_to_obj = torch.norm(obj - tcp, dim=-1)

    in_place_scaling = init_data["block"].in_place_scaling
    obj_to_midpoint = torch.norm((obj - midpoint) * in_place_scaling, dim=-1)
    obj_to_midpoint_init = torch.norm((obj_init_pos - midpoint) * in_place_scaling, dim=-1)

    # obj_to_target = torch.norm(obj - target, dim=-1)
    # obj_to_target_init = torch.norm(obj_init_pos - target, dim=-1)

    in_place_part1 = reward_utils.tolerance(
        obj_to_midpoint,
        bounds=(0, _TARGET_RADIUS),
        margin=obj_to_midpoint_init,
        sigmoid="long_tail",
    )

    # in_place_part2 = reward_utils.tolerance(
    #     obj_to_target,
    #     bounds=(0, _TARGET_RADIUS),
    #     margin=obj_to_target_init,
    #     sigmoid="long_tail",
    # )

    object_reach_radius = 0.01
    obj_radius = 0.015
    pad_success_thresh = 0.05
    xz_thresh = 0.005

    # left_pad_y = _get_left_pad_y(next_obs)
    # right_pad_y = _get_right_pad_y(next_obs)
    left_pad_y = next_obs[:, 1] + next_obs[:, 24]
    right_pad_y = next_obs[:, 1] + next_obs[:, 25]

    object_grasped = reward_utils.gripper_caging_reward(
        left_pad_y,
        right_pad_y,
        act,
        tcp,
        init_tcp,
        obj,
        obj_init_pos,
        object_reach_radius=object_reach_radius,
        obj_radius=obj_radius,
        pad_success_thresh=pad_success_thresh,
        xz_thresh=xz_thresh,
        desired_gripper_effort=0.8,
        high_density=True,
    )
    reward = 4 * object_grasped
    reward = torch.where(
        (tcp_to_obj < 0.02) * (tcp_opened > 0),
        4 * object_grasped + 1.0 + 4.0 * in_place_part1,
        reward,
    )
    reward = torch.where(
        obj_to_midpoint < _TARGET_RADIUS,
        10 * torch.ones_like(reward),
        reward,
    )
    reward = torch.where(
        torch.norm(obj[:, :2], dim=-1) > 0.8,
        -10 * torch.ones_like(reward),
        reward,
    )
    return reward.view(-1, 1)
