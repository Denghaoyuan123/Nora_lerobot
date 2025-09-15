# # nora_function.py

from typing import Dict, List, Union
import numpy as np

# ==== LeRobot 导入 ====
from lerobot.robots import make_robot_from_config
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import (
    OpenCVCameraConfig,
    Cv2Rotation,   # ✅ 引入枚举
)


# ========== 1) 创建并连接 so101_follower ==========
def create_so101_robot(
    port: str = "/dev/ttyACM5",
    robot_id: str = "so101_follower",
    front_index_or_path: int | str = 0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    use_degrees: bool = False,
    max_relative_target: float | None = None,
    disable_torque_on_disconnect: bool = True,
):
    """
    创建并连接 so101_follower 机器人，挂载一个 front OpenCV 相机。
    """

    # 相机配置 (OpenCVCameraConfig)
    cameras = {
        "front": OpenCVCameraConfig(
            index_or_path=front_index_or_path,
            width=width,
            height=height,
            fps=fps,
            color_mode="rgb",   # dataclass 里确实有这个
            rotation=Cv2Rotation.NO_ROTATION,
            warmup_s=0.0,
        )
    }

    # 机器人配置 (SO101FollowerConfig)
    cfg = SO101FollowerConfig(
        id=robot_id,
        port=port,
        cameras=cameras,
        use_degrees=use_degrees,
        max_relative_target=max_relative_target,
        disable_torque_on_disconnect=disable_torque_on_disconnect,
        calibration_dir=None,
        record=False,
        debug=False,
    )

    robot = make_robot_from_config(cfg)
    robot.connect()
    return robot


# ========== 辅助函数：查询 action 名称顺序 ==========
def get_action_names(robot) -> List[str]:
    """返回机器人 action 的顺序名称"""
    feats = robot.action_features
    if isinstance(feats, dict):
        return list(feats.keys())
    return list(feats)


# ========== 2) 关节读数 ==========
def get_joint_positions(robot) -> np.ndarray:
    """
    返回机械臂关节读数（np.ndarray），按 action_features 顺序排列。
    """
    obs = robot.get_observation()
    names = get_action_names(robot)
    vals = []
    missing = []
    for k in names:
        if k in obs:
            vals.append(float(obs[k]))
        else:
            missing.append(k)
    if missing:
        raise KeyError(f"缺少关节键 {missing}；obs.keys()={list(obs.keys())[:10]}...")
    return np.asarray(vals, dtype=np.float32)


# ========== 3) front 图像 ==========
def get_front_image(robot) -> np.ndarray:
    """
    返回 front 相机画面（H×W×3，RGB）。
    """
    obs = robot.get_observation()
    if "front" in obs:
        return np.asarray(obs["front"])
    if "images" in obs and "front" in obs["images"]:
        return np.asarray(obs["images"]["front"])
    raise KeyError(f"obs 中没有 front 图像；obs.keys()={list(obs.keys())[:10]}...")


# ========== 4) 发送关节控制 ==========
def send_joint(
    robot,
    action: Union[np.ndarray, Dict[str, float]],
    clip: bool = True,
):
    """
    下发关节控制:
    - ndarray: 按 action_features 顺序映射到 dict
    - dict: 直接透传
    """
    if isinstance(action, dict):
        act_dict = {k: float(v) for k, v in action.items()}
    else:
        action = np.asarray(action).reshape(-1)
        names = get_action_names(robot)
        if action.shape[0] != len(names):
            raise ValueError(
                f"action 维度不匹配，给了 {action.shape[0]}，但 robot 期望 {len(names)}，顺序为 {names}"
            )
        act_dict = {n: float(v) for n, v in zip(names, action)}

    if clip:
        for k in act_dict:
            act_dict[k] = float(np.clip(act_dict[k], -1.0, 1.0))

    robot.send_action(act_dict)





###jiang_old_version
# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Union
# import numpy as np

# from types import SimpleNamespace
# from lerobot.robots import make_robot_from_config
# from lerobot.cameras.utils import get_cv2_backend  # ✅ 用于设定 OpenCV 后端

# def create_so101_robot(
#     port: str = "/dev/ttyACM5",
#     robot_id: str = "so101_follower",
#     use_opencv_front: bool = True,
#     front_index_or_path: int | str = 0,
#     width: int = 640,
#     height: int = 480,
#     fps: int = 30,
#     use_degrees: bool = False,
#     max_relative_target: float | None = None,
#     disable_torque_on_disconnect: bool = True,
# ):
#     # 1) 相机配置：必须是“有属性”的对象，且包含 OpenCVCamera 需要的字段
#     cameras = None
#     if use_opencv_front:
#         cameras = {
#             "front": SimpleNamespace(
#                 # —— OpenCV 相机必备/常见字段 ——
#                 type="opencv",
#                 index_or_path=front_index_or_path,   # 0 / 1 / '/dev/video2' / 路径
#                 width=width,
#                 height=height,
#                 fps=fps,

#                 # 这些在你的报错栈里会被访问（color_mode 起码是必需）
#                 color_mode="rgb",                    # 部分实现会期望 'rgb' 或 'bgr'；这里给 'rgb'
#                 backend=get_cv2_backend(),           # 选择 OpenCV 捕获后端（Linux 下为 CAP_ANY）

#                 # 其他常见可选字段：给默认值以避免 AttributeError
#                 rotation=None,                       # 不旋转；如需 90/180/270，换枚举即可
#                 exposure=None,
#                 gain=None,
#                 brightness=None,
#                 contrast=None,
#                 saturation=None,
#                 hue=None,
#                 auto_reconnect=True,                 # 若你的实现支持自动重连
#                 buffer_size=1,                       # 尝试减小延迟（如实现支持）
#             )
#         }

#     # 2) 机器人配置：duck-typing（属性访问），补齐 so101 会用到的字段
#     cfg = SimpleNamespace(
#         type="so101_follower",
#         id=robot_id,
#         port=port,
#         cameras=cameras,
#         use_camera=bool(cameras),

#         use_degrees=use_degrees,
#         max_relative_target=max_relative_target,
#         disable_torque_on_disconnect=disable_torque_on_disconnect,

#         # Robot 基类常见字段，给默认以避免 AttributeError
#         calibration_dir=None,
#         output_dir=None,
#         log_dir=None,
#         record=False,
#         debug=False,
#         rate_hz=20,
#     )

#     robot = make_robot_from_config(cfg)
#     robot.connect()
#     return robot



# # ========== 查询 action 名称顺序 ==========
# def get_action_names(robot) -> List[str]:
#     """
#     返回机器人 action 的顺序名称，用于把 ndarray 映射成 dict。
#     """
#     feats = robot.action_features
#     if isinstance(feats, dict):
#         return list(feats.keys())
#     # 兜底：有些实现是 list/tuple
#     return list(feats)


# # ========== 1) 关节读数 ==========
# def get_joint_positions(robot) -> np.ndarray:
#     """
#     返回机械臂关节读数（np.ndarray），按 robot.action_features 的顺序排列。
#     """
#     obs = robot.get_observation()
#     names = get_action_names(robot)  # e.g. ["shoulder_pan.pos", ..., "gripper.pos"]
#     vals = []
#     missing = []
#     for k in names:
#         if k in obs:
#             vals.append(float(obs[k]))
#         else:
#             missing.append(k)
#     if missing:
#         raise KeyError(f"以下关节键在 obs 中缺失：{missing}；obs.keys()={list(obs.keys())[:10]}...")
#     return np.asarray(vals, dtype=np.float32)


# # ========== 2) front 图像 ==========
# def get_front_image(robot) -> np.ndarray:
#     obs = robot.get_observation()
#     if "images" in obs:
#         images = obs["images"]
#         if "front" in images:
#             return np.asarray(images["front"])
#         else:
#             raise KeyError(f"images 里没有 'front'，实际相机：{list(images.keys())}")
#     # 有些相机直接在顶层键
#     if "front" in obs:
#         return np.asarray(obs["front"])
#     raise KeyError(f"没有找到 front 图像；obs.keys()={list(obs.keys())[:10]}...")



# # ========== 3) 发送关节控制 ==========
# def send_joint(
#     robot,
#     action: Union[np.ndarray, Dict[str, float]],
#     clip: bool = True,
# ):
#     """
#     将关节控制量发送给 follower：
#     - 如果传入 ndarray：会按 robot.action_features 的顺序映射到 dict。
#     - 如果传入 dict：直接透传。
#     - 默认把值裁剪到 [-1, 1]（多数 teleop/控制链路用的是归一化量）。
#     """
#     if isinstance(action, dict):
#         act_dict = {k: float(v) for k, v in action.items()}
#     else:
#         action = np.asarray(action).reshape(-1)
#         names = get_action_names(robot)
#         if action.shape[0] != len(names):
#             raise ValueError(
#                 f"action 维度不匹配，给了 {action.shape[0]}，但 robot 期望 {len(names)}，顺序为 {names}"
#             )
#         act_dict = {n: float(v) for n, v in zip(names, action)}

#     if clip:
#         for k in act_dict:
#             act_dict[k] = float(np.clip(act_dict[k], -1.0, 1.0))

#     robot.send_action(act_dict)






