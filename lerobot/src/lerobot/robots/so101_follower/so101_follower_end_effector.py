# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from typing import Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.errors import DeviceNotConnectedError
from lerobot.model.kinematics import RobotKinematics
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from . import SO101Follower
from .config_so101_follower import SO101FollowerEndEffectorConfig

logger = logging.getLogger(__name__)

def _rpy_to_matrix(roll, pitch, yaw):
    """RPY -> 3x3 旋转矩阵（X-roll, Y-pitch, Z-yaw；Rz(yaw) @ Ry(pitch) @ Rx(roll)）"""
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    Rx = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx,  cx],
    ], dtype=np.float32)

    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy],
    ], dtype=np.float32)

    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1],
    ], dtype=np.float32)

    return (Rz @ Ry @ Rx).astype(np.float32)


def _maybe_deg_to_rad(r, p, y):
    """若角度绝对值大于 ~π，则认为是度，自动转弧度；否则保留原值。"""
    arr = np.array([r, p, y], dtype=np.float32)
    if np.max(np.abs(arr)) > 3.2:  # 3.2 ~ π 的旁路阈值
        arr = np.deg2rad(arr)
    return float(arr[0]), float(arr[1]), float(arr[2])

class SO101FollowerEndEffector(SO101Follower):
    """
    SO101Follower robot with end-effector space control.

    This robot inherits from SO101Follower but transforms actions from
    end-effector space to joint space before sending them to the motors.
    """

    config_class = SO101FollowerEndEffectorConfig
    name = "so101_follower_end_effector"

    def __init__(self, config: SO101FollowerEndEffectorConfig):
        super().__init__(config)
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        self.config = config

        # Initialize the kinematics module for the so101 robot
        if self.config.urdf_path is None:
            raise ValueError(
                "urdf_path must be provided in the configuration for end-effector control. "
                "Please set urdf_path in your SO101FollowerEndEffectorConfig."
            )

        self.kinematics = RobotKinematics(
            urdf_path=self.config.urdf_path,
            target_frame_name=self.config.target_frame_name,
        )

        # Store the bounds for end-effector position
        self.end_effector_bounds = self.config.end_effector_bounds

        self.current_ee_pos = None
        self.current_joint_pos = None

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }


    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        接受以下两种格式：
        1) 绝对位姿（推荐）：{"x","y","z","rotation"(=roll),"pitch","yaw","gripper"}
        2) 增量位姿（兼容）：{"delta_x","delta_y","delta_z","gripper"}  # 只改位置，姿态保持不变

        - 位置 (x,y,z) 单位：米（若你是其它单位，请在上游统一换算）
        - 姿态 (r,p,y)：自动识别弧度/角度
        - gripper：建议范围[0,2]，1为中性；内部按 (g-1)*max_gripper_pos 映射到舵机行程
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # 读当前关节与末端位姿缓存
        if self.current_joint_pos is None:
            curr = self.bus.sync_read("Present_Position")
            self.current_joint_pos = np.array([curr[name] for name in self.bus.motors], dtype=np.float64)

        if self.current_ee_pos is None:
            self.current_ee_pos = self.kinematics.forward_kinematics(self.current_joint_pos.astype(np.float64))

        print("self.current_ee_pos:", self.current_ee_pos[:3, 3])

        # ---------- 解析输入 ----------
        # 默认保持当前朝向
        target_R = self.current_ee_pos[:3, :3].copy()
        target_xyz = self.current_ee_pos[:3, 3].copy()
        g = float(action.get("gripper", 1.0))  # 缺省中性

        has_abs_xyz = all(k in action for k in ("x", "y", "z"))
        has_abs_rpy = (("rotation" in action) or ("roll" in action)) and ("pitch" in action) and ("yaw" in action)
        
        if has_abs_xyz:
            # 绝对位置
            target_xyz = np.array([action["x"], action["y"], action["z"]], dtype=np.float32)

        if has_abs_rpy:
            # 绝对姿态
            roll = float(action.get("rotation", action.get("roll", 0.0)))
            pitch = float(action["pitch"])
            yaw = float(action["yaw"])
            roll, pitch, yaw = _maybe_deg_to_rad(roll, pitch, yaw)
            target_R = _rpy_to_matrix(roll, pitch, yaw)


        # ---------- 位置限幅 ----------
        # if self.end_effector_bounds is not None:
        #     target_xyz = np.clip(
        #         target_xyz,
        #         self.end_effector_bounds["min"],
        #         self.end_effector_bounds["max"],
        #     )

        # ---------- 组装 4x4 目标位姿 ----------
        desired_ee_pos = np.eye(4, dtype=np.float32)
        desired_ee_pos[:3, :3] = target_R
        desired_ee_pos[:3, 3] = target_xyz

        # ---------- IK 计算 ----------
        target_joint_values_in_degrees = self.kinematics.inverse_kinematics(
            self.current_joint_pos.astype(np.float64), desired_ee_pos.astype(np.float64)
        )
        print("Target joint values (deg):", target_joint_values_in_degrees)
        # ---------- 关节动作字典 ----------
        joint_action = {
            f"{name}.pos": target_joint_values_in_degrees[i]
            for i, name in enumerate(self.bus.motors.keys())
        }

        # joint_action["gripper.pos"] = float(np.clip(
        #     self.current_joint_pos[-1] + (g - 1.0) * self.config.max_gripper_pos,
        #     5.0,  # 保留你之前的下限
        #     self.config.max_gripper_pos,
        # ))
        g = float(g)
        joint_action["gripper.pos"] = np.clip(g * 100, 5.0, 50)
        print(joint_action["gripper.pos"])
        # breakpoint()
        # ---------- 更新缓存并下发 ----------
        self.current_ee_pos = desired_ee_pos.copy()
        self.current_joint_pos = target_joint_values_in_degrees.astype(np.float32).copy()
        self.current_joint_pos[-1] = joint_action["gripper.pos"]

        return super().send_action(joint_action)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def reset(self):
        self.current_ee_pos = None
        self.current_joint_pos = None
