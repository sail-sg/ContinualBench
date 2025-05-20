from typing import NamedTuple

import mujoco
import numpy as np
import torch
import tree
from gym.spaces import Box

from continual_bench.envs import reward_fns, reward_utils, torch_reward_utils
from continual_bench.envs.asset_path_utils import full_v2_path_for
from continual_bench.envs.mujoco.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class ButtonInit(NamedTuple):
    obj_init_pos: np.ndarray
    target_pos: np.ndarray
    obj_to_target_init: np.ndarray


class DoorInit(NamedTuple):
    obj_init_pos: np.ndarray
    target_pos: np.ndarray


class WindowInit(NamedTuple):
    obj_init_pos: np.ndarray
    target_pos: np.ndarray
    window_handle_pos_init: np.ndarray


class FaucetInit(NamedTuple):
    obj_init_pos: np.ndarray
    target_pos: np.ndarray


class PegInit(NamedTuple):
    obj_init_pos: np.ndarray
    target_pos: np.ndarray


class BlockInit(NamedTuple):
    obj_init_pos: np.ndarray
    target_pos: np.ndarray
    midpoint: np.ndarray
    in_place_scaling: np.ndarray


class TaskSpec(NamedTuple):
    name: str
    hand_init_pos: np.ndarray
    obj_init_pos: np.ndarray


class SawyerBenchEnv(SawyerXYZEnv):
    all_tasks = [
        "button",
        "door",
        "window",
        "faucet",
        "peg",
        "block",
    ]
    hand_init_spaces = {
        "button": Box(
            low=np.array((0, -0.3, 0.1)),
            high=np.array((0, -0.3, 0.1)),
        ),
        "door": Box(
            low=np.array((0, 0.6, 0.2)),
            high=np.array((0, 0.6, 0.2)),
        ),
        "window": Box(
            low=np.array((-0.2, 0.3, 0.1)),
            high=np.array((-0.2, 0.3, 0.1)),
        ),
        "faucet": Box(
            low=np.array((0, -0.3, 0.1)),
            high=np.array((0, -0.3, 0.1)),
        ),
        "peg": Box(
            low=np.array((0.15, -0.75, 0.2)),
            high=np.array((0.15, -0.75, 0.2)),
        ),
        "block": Box(
            low=np.array((0.25, 0.1, 0.2)),
            high=np.array((0.25, 0.1, 0.2)),
        ),
    }
    obj_init_positions = {
        "button": np.array((-0.5, -0.5, 0)),
        "door": np.array((0.05, 0.9, 0.15)),
        "window": np.array((-0.7, 0.2, 0.202)),
        "faucet": np.array((0.6, -0.4, 0.0)),
        "peg": np.array((0, -0.75, 0)),
        "block": np.array((0.5, 0.2, 0)),
    }

    def __init__(self, tasks=None, render_mode=None):
        obj_low = (-0.56, 0.35, 0.115)
        obj_high = (-0.56, 0.35, 0.115)

        super().__init__(
            self.model_name,
            hand_low=[-0.8, -0.8, 0.05],
            hand_high=[+0.8, 0.8, 0.5],
            render_mode=render_mode,
        )
        if tasks is not None:
            self.tasks = tasks

        # self.init_config = {
        #     "obj_init_pos": np.array([0, 0.8, 0.115], dtype=np.float32),
        #     "hand_init_pos": np.array([0, 0.4, 0.2], dtype=np.float32),
        # }
        self.goal = np.array([0, 0.88, 0.1])
        # self.obj_init_pos = self.init_config["obj_init_pos"]
        # self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.init_data = {}
        self.set_task(self.all_tasks[0])
        self._prev_gripper = None

    def set_task(self, task_name: str):
        assert task_name in self.all_tasks, f"{task_name} not in {self.all_tasks}"
        self.task_spec = TaskSpec(
            name=task_name,
            hand_init_pos=self.hand_init_spaces[task_name].sample(),
            obj_init_pos=self.obj_init_positions[task_name],
        )
        self.camera_name = f"camera_{task_name}"

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_bench.xml")

    def reset(self, seed=None, options=None):
        self.curr_path_length = 0
        self._reset_simulation()
        obs = self.reset_model()
        info = self._get_reset_info()

        if self.render_mode == "human":
            self.render()
        return obs

    def reset_model(self):
        # Hand.
        self.hand_init_pos = self.task_spec.hand_init_pos
        self._reset_hand()

        # Objects.
        self.obj_init_pos = self.task_spec.obj_init_pos

        # 1) Button.
        obj_init_pos = self.obj_init_positions["button"]
        self.model.body("button1").pos = obj_init_pos
        # self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "button1")] = obj_init_pos
        # mujoco.mj_forward(self.model, self.data)
        self._set_pos_site("button1", obj_init_pos)
        target_pos = self._get_site_pos("buttonHole")
        obj_to_target_init = abs(target_pos[2] - self._get_site_pos("buttonStart")[2])
        self.init_data["button"] = ButtonInit(
            obj_init_pos=obj_init_pos, target_pos=target_pos, obj_to_target_init=obj_to_target_init
        )

        # 2) Door.
        obj_init_pos = self.obj_init_positions["door"]  # TODO DEFINE ME.
        target_pos = obj_init_pos + np.array([-0.3, -0.45, 0.0])
        # self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "door")] = obj_init_pos
        self.model.body("door").pos = obj_init_pos
        self.model.site("doorGoal").pos = target_pos
        self.init_data["door"] = DoorInit(obj_init_pos=obj_init_pos, target_pos=target_pos)

        # 3) Window.
        obj_init_pos = self.obj_init_positions["window"]  # TODO DEFINE ME.
        target_pos = obj_init_pos.copy()
        self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "window")] = obj_init_pos
        window_handle_pos_init = self._get_site_pos("handleCloseStart") + np.array([0.0, 0.2, 0.0])
        self.data.joint("window_slide").qpos = 0.2
        self.init_data["window"] = WindowInit(
            obj_init_pos=obj_init_pos,
            target_pos=target_pos,
            window_handle_pos_init=window_handle_pos_init,
        )

        # 4) Faucet.
        obj_init_pos = self.obj_init_positions["faucet"]
        handle_length = 0.175
        target_pos = obj_init_pos + np.array([handle_length, 0.0, 0.125])
        self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "faucet")] = obj_init_pos
        self.model.site("faucetGoal").pos = target_pos
        self.init_data["faucet"] = FaucetInit(obj_init_pos=obj_init_pos, target_pos=target_pos)

        # 5) Peg.
        box_init_pos = self.obj_init_positions["peg"]
        self.model.body_pos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pegBox")] = box_init_pos
        pos_plug = box_init_pos + np.array([0.044, 0.0, 0.131])
        _qpos = self.data.qpos.flat.copy()
        _qvel = self.data.qvel.flat.copy()
        _qpos[13:16] = pos_plug
        _qpos[16:20] = np.array([1.0, 0.0, 0.0, 0.0])
        _qvel[13:16] = 0
        self.set_state(_qpos, _qvel)

        obj_init_pos = self._get_site_pos("pegEnd")
        target_pos = pos_plug + np.array([0.15, 0.0, 0.0])
        self.model.site("pegGoal").pos = target_pos
        if self.task_spec.name == "peg":
            self.obj_init_pos = obj_init_pos
        self.init_data["peg"] = PegInit(obj_init_pos=obj_init_pos, target_pos=target_pos)

        # 6) Block.
        block_pos = self.obj_init_positions["block"]
        _qpos = self.data.qpos.flat.copy()
        _qvel = self.data.qvel.flat.copy()
        _qpos[20:23] = block_pos
        _qvel[20:26] = 0
        self.set_state(_qpos, _qvel)
        self.init_data["block"] = BlockInit(
            obj_init_pos=block_pos,
            target_pos=self._get_site_pos("blockGoal"),
            midpoint=np.array((0.52, 0.33, 0.2)),
            in_place_scaling=np.array([1.0, 1.0, 1.0]),
        )

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs()

    def _get_obs(self):
        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self.data.body("rightclaw"),
            self.data.body("leftclaw"),
        )
        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco

        gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

        if self._prev_gripper is None:
            self._prev_gripper = gripper_distance_apart
        gripper_direction = gripper_distance_apart - self._prev_gripper
        self._prev_gripper = gripper_distance_apart

        left_pad_y_diff = self.get_body_com("leftpad")[1] - pos_hand[1]
        right_pad_y_diff = self.get_body_com("rightpad")[1] - pos_hand[1]

        # Button.
        button_pos = self.get_body_com("button") + np.array([0.0, 0.0, 0.193])
        # button_target_z = self.init_data["button"].target_pos[2]

        # Door.
        door_handle_pos = self.data.geom("handle").xpos.copy()
        door_handle_angle = self.data.joint("doorjoint").qpos
        # door_target_x = self.init_data["door"].target_pos[0]

        # Window.
        window_handle_pos = self._get_site_pos("windowHandleCloseStart")

        # Faucet.
        faucet_handle_pos = self._get_site_pos("faucetHandleStartClose") + np.array([0.0, 0.0, -0.01])

        # Peg.
        peg_pos = self._get_site_pos("pegEnd")

        # Block.
        block_pos = self.get_body_com("block")

        return np.hstack(
            (
                pos_hand,
                gripper_distance_apart,
                button_pos,
                # button_target_z,
                door_handle_pos,
                door_handle_angle,
                # door_target_x,
                window_handle_pos,
                faucet_handle_pos,
                peg_pos,
                block_pos,
                gripper_direction,
                left_pad_y_diff,
                right_pad_y_diff,
            )
        )

    @property
    def sawyer_observation_space(self):
        gripper_low = 0.2
        gripper_high = 1.0

        button_low = np.array((-1, -1, 0.1))
        button_high = np.array((1, 1, 0.3))

        door_low = np.array((-0.3, 0.3, 0, -1.7))
        door_high = np.array((0.4, 0.8, 0.3, 0.1))

        window_low = np.array((-0.8, 0.2, 0))
        window_high = np.array((-0.4, 0.5, 0.4))

        faucet_low = (self.obj_init_positions["faucet"][:2] - np.array((0.2, 0.2))).tolist() + [0]
        faucet_high = (self.obj_init_positions["faucet"][:2] + np.array((0.2, 0.2))).tolist() + [0.228]

        peg_low = np.array((0, -0.8, 0))
        peg_high = np.array((0.2, 0.8, 0.5))

        block_low = np.array((-0.8, -0.8, 0))
        block_high = np.array((0.8, 0.8, 0.5))

        gripper_direction_low = np.array((-0.1,))
        gripper_direction_high = np.array((0.1,))

        pad_y_diff_low = np.array((0, -0.05))
        pad_y_diff_high = np.array((0.05, 0))

        return Box(
            np.hstack(
                (
                    self._HAND_SPACE.low,
                    gripper_low,
                    button_low,
                    door_low,
                    window_low,
                    faucet_low,
                    peg_low,
                    block_low,
                    gripper_direction_low,
                    pad_y_diff_low,
                )
            ),
            np.hstack(
                (
                    self._HAND_SPACE.high,
                    gripper_high,
                    button_high,
                    door_high,
                    window_high,
                    faucet_high,
                    peg_high,
                    block_high,
                    gripper_direction_high,
                    pad_y_diff_high,
                )
            ),
            dtype=np.float32,
        )

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        info = {}
        rewards = {}
        # Button.
        (
            button_reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed,
        ) = self.button_compute_reward(action, obs)
        info["button"] = {
            "success": float(obj_to_target <= 0.024),
            "near_object": float(tcp_to_obj <= 0.05),
            "tcp_open": float(tcp_open > 0),
            "near_button": near_button,
            "button_pressed": button_pressed,
            "obj_to_target": obj_to_target,
            "unscaled_reward": button_reward,
        }
        rewards["button"] = button_reward

        # Door.
        (
            door_reward,
            reward_grab,
            obj_to_target,
            reward_ready,
            reward_success,
        ) = self.door_compute_reward(action, obs)
        info["door"] = {
            "success": float(obj_to_target <= 0.08),
            "near_object": reward_ready,
            "grasp_success": float(reward_grab >= 0.5),
            "grasp_reward": reward_grab,
            "in_place_reward": reward_success,
            "obj_to_target": obj_to_target,
            "unscaled_reward": door_reward,
        }
        rewards["door"] = door_reward

        # Window.
        (
            window_reward,
            tcp_to_obj,
            _,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.window_compute_reward(action, obs)
        info["window"] = {
            "success": float(target_to_obj <= self.TARGET_RADIUS),
            "near_object": float(tcp_to_obj <= 0.05),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": window_reward,
        }
        rewards["window"] = window_reward

        # Faucet.
        (
            faucet_reward,
            tcp_to_obj,
            _,
            target_to_obj,
            object_grasped,
            in_place,
        ) = self.faucet_compute_reward(action, obs)
        info["faucet"] = {
            "success": float(target_to_obj <= 0.07),
            "near_object": float(tcp_to_obj <= 0.01),
            "grasp_success": 1.0,
            "grasp_reward": object_grasped,
            "in_place_reward": in_place,
            "obj_to_target": target_to_obj,
            "unscaled_reward": faucet_reward,
        }
        rewards["faucet"] = faucet_reward

        # Peg.
        (
            peg_reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
            grasp_success,
        ) = self.peg_compute_reward(action, obs)
        info["peg"] = {
            "success": float(obj_to_target <= 0.07),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_success": grasp_success,
            "grasp_reward": grasp_reward,
            "in_place_reward": in_place_reward,
            "obj_to_target": obj_to_target,
            "unscaled_reward": peg_reward,
        }
        rewards["peg"] = peg_reward

        # Block.
        (
            block_reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            gt_grasp_reward,
            debug_own_grasp_fn,
            debug_grasp_reward_tcp,
            debug_grasp_reward_pad,
            hand_pos,
            tcp,
            leftpad,
            rightpad,
        ) = self.block_compute_reward(action, obs)
        info["block"] = {
            "success": float(obj_to_target <= 0.07),
            "near_object": float(tcp_to_obj <= 0.03),
            "grasp_success": float((tcp_open > 0) and (obs[22] - 0.02 > self.obj_init_pos[2])),
            "grasp_reward": grasp_reward,
            "diff_own_grasp_fn": abs(debug_own_grasp_fn - gt_grasp_reward),
            "diff_grasp_reward_tcp": abs(debug_grasp_reward_tcp - gt_grasp_reward),
            "diff_grasp_reward_pad": abs(debug_grasp_reward_pad - gt_grasp_reward),
            "obj_to_target": obj_to_target,
            "unscaled_reward": block_reward,
        }
        rewards["block"] = block_reward

        return rewards, info

    @property
    def _target_site_config(self):
        return []

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id("btnGeom")

    def _get_pos_objects(self):
        return self.get_body_com("button") + np.array([0.0, 0.0, 0.193])

    def _get_quat_objects(self):
        return self.data.body("button").xquat

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def button_compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center
        target_pos = self.init_data["button"].target_pos
        obj_to_target_init = self.init_data["button"].obj_to_target_init

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(target_pos[2] - obj[2])

        tcp_closed = 1 - obs[3]
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
        if tcp_to_obj <= 0.03:
            reward += 5 * button_pressed

        return (reward, tcp_to_obj, obs[3], obj_to_target, near_button, button_pressed)

    @staticmethod
    def _door_reward_grab_effort(action):
        return (np.clip(action[3], -1, 1) + 1.0) / 2.0

    @staticmethod
    def _door_reward_pos(obs, theta):
        hand = obs[:3]
        door = obs[7:10] + np.array([-0.05, 0, 0])

        threshold = 0.12
        # floor is a 3D funnel centered on the door handle
        radius = np.linalg.norm(hand[:2] - door[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = (
            1.0
            if hand[2] >= floor
            else reward_utils.tolerance(
                floor - hand[2],
                bounds=(0.0, 0.01),
                margin=max(0, floor / 2.0),
                sigmoid="long_tail",
            )
        )
        # move the hand to a position between the handle and the main door body
        in_place = reward_utils.tolerance(
            np.linalg.norm(hand - door - np.array([0.05, 0.03, -0.01])),
            bounds=(0, threshold / 2.0),
            margin=0.5,
            sigmoid="long_tail",
        )
        ready_to_open = reward_utils.hamacher_product(above_floor, in_place)

        # now actually open the door
        door_angle = -theta
        a = 0.2  # Relative importance of just *trying* to open the door at all
        b = 0.8  # Relative importance of fully opening the door
        opened = a * float(theta < -np.pi / 90.0) + b * reward_utils.tolerance(
            np.pi / 2.0 + np.pi / 6 - door_angle,
            bounds=(0, 0.5),
            margin=np.pi / 3.0,
            sigmoid="long_tail",
        )

        return ready_to_open, opened

    def door_compute_reward(self, action, obs):
        theta = obs[10]
        target_pos_x = self.init_data["door"].target_pos[0]

        reward_grab = self._door_reward_grab_effort(action)
        reward_steps = self._door_reward_pos(obs, theta)

        reward = sum(
            (
                2.0 * reward_utils.hamacher_product(reward_steps[0], reward_grab),
                8.0 * reward_steps[1],
            )
        )

        # Override reward on success flag
        obj_to_target = abs(obs[7] - target_pos_x)
        if obj_to_target <= 0.08:
            reward = 10.0

        return (
            reward,
            reward_grab,
            obj_to_target,
            *reward_steps,
        )

    def window_compute_reward(self, action, obs):
        del action
        obj = obs[11:14]
        tcp = self.tcp_center
        target = self.init_data["window"].target_pos
        window_handle_pos_init = self.init_data["window"].window_handle_pos_init

        target_to_obj = obj[1] - target[1]
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = window_handle_pos_init[1] - target[1]
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid="long_tail",
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(window_handle_pos_init - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid="gaussian",
        )
        tcp_opened = 0
        object_grasped = reach

        reward = 10 * reward_utils.hamacher_product(reach, in_place)

        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)

    def faucet_compute_reward(self, action, obs):
        target_radius = 0.07
        del action
        obj = obs[14:17]
        tcp = self.tcp_center
        target = self.init_data["faucet"].target_pos
        obj_init_pos = self.init_data["faucet"].obj_init_pos

        target_to_obj = obj - target
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = obj_init_pos - target
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, target_radius),
            margin=abs(target_to_obj_init - target_radius),
            sigmoid="long_tail",
        )

        faucet_reach_radius = 0.01
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, faucet_reach_radius),
            margin=abs(tcp_to_obj_init - faucet_reach_radius),
            sigmoid="gaussian",
        )

        tcp_opened = 0
        object_grasped = reach

        reward = 2 * reach + 3 * in_place
        reward *= 2
        reward = 10 if target_to_obj <= target_radius else reward

        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped, in_place)

    def peg_compute_reward(self, action, obs):
        tcp = self.tcp_center
        obj = obs[17:20]
        tcp_opened = obs[3]
        target = self.init_data["peg"].target_pos
        obj_init_pos = self.init_data["peg"].obj_init_pos

        tcp_to_obj = np.linalg.norm(obj - tcp)
        obj_to_target = np.linalg.norm(obj - target)
        pad_success_margin = 0.05
        object_reach_radius = 0.01
        x_z_margin = 0.005
        obj_radius = 0.025

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=object_reach_radius,
            obj_radius=obj_radius,
            pad_success_thresh=pad_success_margin,
            xz_thresh=x_z_margin,
            desired_gripper_effort=0.8,
            high_density=True,
        )

        in_place_margin = np.linalg.norm(obj_init_pos - target)

        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.05),
            margin=in_place_margin,
            sigmoid="long_tail",
        )
        grasp_success = tcp_opened > 0.5 and (obj[0] - obj_init_pos[0] > 0.015)

        reward = 2 * object_grasped

        if grasp_success and tcp_to_obj < 0.035:
            reward = 1 + 2 * object_grasped + 5 * in_place

        if obj_to_target <= 0.05:
            reward = 10.0

        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place,
            float(grasp_success),
        )

    def block_compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        obj = obs[20:23]
        tcp_opened = obs[3]
        # target = self.init_data["block"].target_pos
        obj_init_pos = self.init_data["block"].obj_init_pos

        midpoint = self.init_data["block"].target_pos

        in_place_scaling = self.init_data["block"].in_place_scaling
        obj_to_midpoint = np.linalg.norm((obj - midpoint) * in_place_scaling)
        obj_to_midpoint_init = np.linalg.norm((obj_init_pos - midpoint) * in_place_scaling)

        # obj_to_target = np.linalg.norm(obj - target)
        # obj_to_target_init = np.linalg.norm(obj_init_pos - target)

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

        # left_pad_y = reward_fns._get_left_pad_y(torch.from_numpy(obs[None]))
        # right_pad_y = reward_fns._get_right_pad_y(torch.from_numpy(obs[None]))
        left_pad_y = obs[1] + obs[24]
        right_pad_y = obs[1] + obs[25]
        left_pad_y = torch.from_numpy(left_pad_y.astype(np.float32)[None])
        right_pad_y = torch.from_numpy(right_pad_y.astype(np.float32)[None])
        tcp = reward_fns._get_tcp_center(torch.from_numpy(obs[None]))

        gt_grasp_reward = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=object_reach_radius,
            obj_radius=obj_radius,
            pad_success_thresh=pad_success_thresh,
            xz_thresh=xz_thresh,
            desired_gripper_effort=0.8,
            high_density=True,
        )
        gt_left_pad_y = torch.from_numpy(self.get_body_com("leftpad")[1][None])
        gt_right_pad_y = torch.from_numpy(self.get_body_com("rightpad")[1][None])
        gt_tcp = torch.from_numpy(self.tcp_center[None])

        debug_own_grasp_fn = torch_reward_utils.gripper_caging_reward(
            gt_left_pad_y,
            gt_right_pad_y,
            torch.from_numpy(action[None]),
            gt_tcp,
            torch.from_numpy(self.init_tcp[None]),
            torch.from_numpy(obj[None]),
            torch.from_numpy(obj_init_pos[None]),
            object_reach_radius=object_reach_radius,
            obj_radius=obj_radius,
            pad_success_thresh=pad_success_thresh,
            xz_thresh=xz_thresh,
            desired_gripper_effort=0.8,
            high_density=True,
        ).numpy()[0]

        debug_grasp_reward_tcp = torch_reward_utils.gripper_caging_reward(
            gt_left_pad_y,
            gt_right_pad_y,
            torch.from_numpy(action[None]),
            tcp,
            torch.from_numpy(self.init_tcp[None]),
            torch.from_numpy(obj[None]),
            torch.from_numpy(obj_init_pos[None]),
            object_reach_radius=object_reach_radius,
            obj_radius=obj_radius,
            pad_success_thresh=pad_success_thresh,
            xz_thresh=xz_thresh,
            desired_gripper_effort=0.8,
            high_density=True,
        ).numpy()[0]

        # debug_grasp_reward_pad = torch_reward_utils.gripper_caging_reward(
        #     left_pad_y,
        #     right_pad_y,
        #     torch.from_numpy(action[None]),
        #     gt_tcp,
        #     torch.from_numpy(self.init_tcp[None]),
        #     torch.from_numpy(obj[None]),
        #     torch.from_numpy(obj_init_pos[None]),
        #     object_reach_radius=object_reach_radius,
        #     obj_radius=obj_radius,
        #     pad_success_thresh=pad_success_thresh,
        #     xz_thresh=xz_thresh,
        #     desired_gripper_effort=0.8,
        #     high_density=True,
        # ).numpy()[0]

        object_grasped = torch_reward_utils.gripper_caging_reward(
            left_pad_y,
            right_pad_y,
            torch.from_numpy(action[None]),
            tcp,
            torch.from_numpy(self.init_tcp[None]),
            torch.from_numpy(obj[None]),
            torch.from_numpy(obj_init_pos[None]),
            object_reach_radius=object_reach_radius,
            obj_radius=obj_radius,
            pad_success_thresh=pad_success_thresh,
            xz_thresh=xz_thresh,
            desired_gripper_effort=0.8,
            high_density=True,
        ).numpy()[0]
        reward = 4 * object_grasped

        tcp_to_obj = np.linalg.norm(obj - tcp[0].numpy())

        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward = 4 * object_grasped + 1.0 + 4.0 * in_place_part1
            # if obj[1] > 0.33:
            #     reward = 2 * object_grasped + 1.0 + 4.0 + 3.0 * in_place_part2

        if obj_to_midpoint < _TARGET_RADIUS:
            reward = 10.0

        if np.linalg.norm(obj[:2]) > 0.8:
            reward = -10.0

        return [
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_midpoint,
            object_grasped,
            gt_grasp_reward,
            debug_own_grasp_fn,
            debug_grasp_reward_tcp,
            debug_grasp_reward_pad,
            self.get_endeff_pos(),
            self.tcp_center,
            self.get_body_com("leftpad"),
            self.get_body_com("rightpad"),
        ]

    def get_init_state(self):
        init_state = {
            "init_data": self.init_data,
            "init_tcp": self.init_tcp,
        }

        def expand_axis0(x):
            if isinstance(x, np.ndarray):
                return x[None, ...]
            return x

        return tree.map_structure(expand_axis0, init_state)
