#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Nora inference on a real LeRobot robot (so101_follower) and record episodes.
- Coerces camera config dicts into dataclass configs (OpenCV/RealSense)
- Robustly converts camera frames to HxWx3 uint8 before feeding Nora
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# ===== Nora & Unnormalizer (来自你的实现) =====
from nora_inference_on_dataset import Nora, build_lerobot_unnormalizer

from lerobot.robots.so101_follower.so101_follower_end_effector import SO101FollowerEndEffector, SO101FollowerEndEffectorConfig

# ===== LeRobot essentials =====
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots import make_robot_from_config, so101_follower
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

# Camera configs for coercion
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


# -------------------------------
# Utils
# -------------------------------
def to_torch_dtype(dtype_str: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]


def normalize_gripper_to_sign(act: np.ndarray) -> np.ndarray:
    out = act.copy()
    if out.ndim == 2 and out.shape[0] == 1:
        out = out[0]
    if out.shape[-1] >= 7:
        out[-1] = np.sign(out[-1])
    return out


def invert_gripper(act: np.ndarray) -> np.ndarray:
    out = act.copy()
    if out.ndim == 2 and out.shape[0] == 1:
        out = out[0]
    if out.shape[-1] >= 7:
        out[-1] = -out[-1]
    return out


def _coerce_camera_config(name: str, cam_dict: dict):
    """
    将普通 dict 转成相机配置 dataclass.
    支持:
      - {"type":"opencv","index_or_path":..., "width":..., "height":..., "fps":...}
      - {"type":"realsense","serial_number_or_name":..., "width":..., "height":..., "fps":...}
    兼容 "camera_index" 作为 "index_or_path" 的别名。
    """
    cam_type = str(cam_dict.get("type", "")).lower()
    if cam_type == "opencv":
        idx = cam_dict.get("index_or_path", cam_dict.get("camera_index", 0))
        return OpenCVCameraConfig(
            index_or_path=idx,
            width=int(cam_dict["width"]),
            height=int(cam_dict["height"]),
            fps=int(cam_dict["fps"]),
        )
    elif cam_type == "realsense":
        return RealSenseCameraConfig(
            serial_number_or_name=cam_dict.get("serial_number_or_name", cam_dict.get("index_or_path")),
            width=int(cam_dict["width"]),
            height=int(cam_dict["height"]),
            fps=int(cam_dict["fps"]),
        )
    else:
        raise ValueError(f"Camera '{name}': unknown type '{cam_type}'. Expected 'opencv' or 'realsense'.")


def _to_hwc_uint8(img, expected_w=None, expected_h=None):
    """把各种形状/类型的图像统一成 H×W×C (uint8, C=3)"""
    # torch -> numpy
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()

    img = np.asarray(img)

    # squeeze 掉前面的 size=1 维度
    while img.ndim > 2 and img.shape[0] == 1:
        img = np.squeeze(img, axis=0)

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.moveaxis(img, 0, -1)  # C,H,W -> H,W,C

    # 灰度 -> 3 通道
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)

    # 尝试按期望分辨率复原异常形状
    if expected_w and expected_h:
        H, W = expected_h, expected_w
        if img.ndim != 3 or img.shape[0] != H or img.shape[1] != W:
            flat = img.reshape(-1)
            if flat.size >= H * W * 3:
                img = flat[: H * W * 3].reshape(H, W, 3)
            elif flat.size >= H * W:
                g = flat[: H * W].reshape(H, W)
                img = np.stack([g] * 3, axis=-1)

    # dtype -> uint8
    if img.dtype != np.uint8:
        if img.dtype in (np.float16, np.float32, np.float64) and img.size and float(img.max()) <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    # 最终保证 H×W×3
    if img.ndim != 3 or img.shape[-1] not in (3, 4):
        img = img.reshape(img.shape[0], img.shape[1], -1)
        if img.shape[-1] == 4:
            img = img[..., :3]
        elif img.shape[-1] != 3:
            need = 3 - img.shape[-1]
            if need > 0:
                img = np.concatenate([img] + [img[..., :1]] * need, axis=-1)
            else:
                img = img[..., :3]
    if img.shape[-1] == 4:
        img = img[..., :3]

    return img


# -------------------------------
# Builders
# -------------------------------
def build_robot(args):
    # 使用末端控制版本
    if args.__dict__["robot.type"] not in ("so101_follower_end_effector", "so101_ee"):
        raise NotImplementedError(
            "请设置 --robot.type=so101_follower_end_effector（或 so101_ee 的别名）以启用末端控制。"
        )

    raw = json.loads(args.__dict__["robot.cameras"])
    cameras = {name: _coerce_camera_config(name, cfg) for name, cfg in raw.items()}

    cfg = SO101FollowerEndEffectorConfig(
        port=args.__dict__["robot.port"],
        cameras=cameras,
        id=args.__dict__["robot.id"],

        # 下面这些参数按真实路径/标定填写
        urdf_path=args.__dict__["robot.urdf_path"],
        target_frame_name=args.__dict__["robot.target_frame"],
        end_effector_step_sizes={"x": args.__dict__["robot.ee_step_x"],
                                 "y": args.__dict__["robot.ee_step_y"],
                                 "z": args.__dict__["robot.ee_step_z"]},
        end_effector_bounds={
            "min": np.array([args.__dict__["robot.ee_min_x"],
                             args.__dict__["robot.ee_min_y"],
                             args.__dict__["robot.ee_min_z"]], dtype=np.float32),
            "max": np.array([args.__dict__["robot.ee_max_x"],
                             args.__dict__["robot.ee_max_y"],
                             args.__dict__["robot.ee_max_z"]], dtype=np.float32),
        },
        max_gripper_pos=args.__dict__["robot.max_gripper_pos"],
    )
    robot = SO101FollowerEndEffector(cfg)
    return robot


def create_or_resume_dataset(args, robot):
    fps = args.__dict__["dataset.fps"]
    ds_root = args.__dict__["dataset.root"]
    repo_id = args.__dict__["dataset.repo_id"]
    use_videos = args.__dict__["dataset.video"]

    action_features = hw_to_dataset_features(robot.action_features, "action", use_videos)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_videos)
    dataset_features = {**action_features, **obs_features}

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=ds_root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=use_videos,
        image_writer_processes=args.__dict__["dataset.num_image_writer_processes"],
        image_writer_threads=args.__dict__["dataset.num_image_writer_threads_per_camera"] * len(getattr(robot, "cameras", {})),
        batch_encoding_size=args.__dict__["dataset.video_encoding_batch_size"],
    )

    if hasattr(robot, "cameras") and len(robot.cameras) > 0:
        ds.start_image_writer(
            num_processes=args.__dict__["dataset.num_image_writer_processes"],
            num_threads=args.__dict__["dataset.num_image_writer_threads_per_camera"] * len(robot.cameras),
        )
    return ds


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser("Nora on real LeRobot (so101_follower)")

    # Robot
    p.add_argument("--robot.type", type=str, default="so101_follower_end_effector")
    p.add_argument("--robot.port", type=str, required=True)
    p.add_argument("--robot.cameras", type=str, required=True,
                   help="JSON string (use outer single quotes, inner double quotes)")
    p.add_argument("--robot.id", type=str, default="my_awesome_follower_arm")
    
    p.add_argument("--robot.urdf_path", type=str, default="/home/luka/Nora_lerobot/lerobot/src/lerobot/robots/so101_follower/SO101/so101_new_calib.urdf")
    p.add_argument("--robot.target_frame", type=str, default="gripper_frame_link")
    p.add_argument("--robot.ee_step_x", type=float, default=0.002)
    p.add_argument("--robot.ee_step_y", type=float, default=0.002)
    p.add_argument("--robot.ee_step_z", type=float, default=0.002)
    p.add_argument("--robot.ee_min_x", type=float, default=0.05)
    p.add_argument("--robot.ee_min_y", type=float, default=-0.20)
    p.add_argument("--robot.ee_min_z", type=float, default=0.02)
    p.add_argument("--robot.ee_max_x", type=float, default=0.35)
    p.add_argument("--robot.ee_max_y", type=float, default=0.20)
    p.add_argument("--robot.ee_max_z", type=float, default=0.10)
    p.add_argument("--robot.max_gripper_pos", type=float, default=100.0)

    # Dataset/Recording
    p.add_argument("--dataset.repo_id", type=str, default="xuanyuanj/eval_nora_so101_pick_the_cube")
    p.add_argument("--dataset.single_task", type=str, default="so101_pick_the_cube")
    p.add_argument("--dataset.root", type=str, default="/home/luka/Nora_lerobot/so101_pick_the_cube_ee")
    p.add_argument("--dataset.fps", type=int, default=15)
    p.add_argument("--dataset.num_episodes", type=int, default=5)
    p.add_argument("--dataset.episode_time_s", type=float, default=20)
    p.add_argument("--dataset.reset_time_s", type=float, default=10)
    p.add_argument("--dataset.video", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dataset.push_to_hub", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dataset.private", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dataset.num_image_writer_processes", type=int, default=0)
    p.add_argument("--dataset.num_image_writer_threads_per_camera", type=int, default=4)
    p.add_argument("--dataset.video_encoding_batch_size", type=int, default=1)

    # Viz
    p.add_argument("--display_data", action=argparse.BooleanOptionalAction, default=False)

    # Nora
    p.add_argument("--model_path", type=str, default="/home/luka/Nora_lerobot/lerobot_training/hf_model_40000")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--instruction", type=str, default="pick up the green cube.")

    # Unnormalize
    p.add_argument("--lerobot_dataset_root", type=str, default="/home/luka/Nora_lerobot/so101_pick_the_cube_ee",
                   help="Root for building Unnormalize. If None, uses dataset.root or ~/.cache/lerobot")
    p.add_argument("--dataset_for_unnorm", type=str, default="xuanyuanj/eval_nora_so101_pick_the_cube",
                   help="HF repo_id (training dataset) to build unnormalizer; default uses dataset.repo_id")

    # Camera selection for Nora input
    p.add_argument("--camera_for_nora", type=str, default="front",
                   help="Which camera name to feed Nora (must match cameras dict key)")

    # Optional post-process
    p.add_argument("--map_gripper_to_neg1_pos1", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--invert_gripper", action=argparse.BooleanOptionalAction, default=False)

    return p.parse_args()


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()

    # Init Nora
    dtype = to_torch_dtype(args.dtype)
    print("[Nora] Loading model:", args.model_path)
    nora = Nora(model_path=args.model_path, device=args.device, torch_dtype=dtype)

    # Build unnormalizer from dataset meta (LeRobot)
    unnorm_repo = args.dataset_for_unnorm
    lerobot_root = args.lerobot_dataset_root 
    unnormalize = build_lerobot_unnormalizer(unnorm_repo, lerobot_root)

    # Robot & dataset
    robot = build_robot(args)
    

    single_task = args.__dict__["dataset.single_task"]
    fps = args.__dict__["dataset.fps"]
    episode_time_s = args.__dict__["dataset.episode_time_s"]
    reset_time_s = args.__dict__["dataset.reset_time_s"]

    if args.display_data:
        _init_rerun(session_name="nora_recording")

    # Connect & keyboard
    robot.connect()
    listener, events = init_keyboard_listener()

    cam_name = args.camera_for_nora
    # 期望分辨率，用于修复异常形状
    expected_w = robot.cameras[cam_name].width
    expected_h = robot.cameras[cam_name].height

    init_action = {
        "x": 0.1487499475479126,
        "y": 0.001560,
        "z": 0.03,
        "rotation": 3.1415903,
        "pitch": 0.04194974899291992,
        "yaw": -3.14146447,
        "gripper": 0.02276399,
    }
    # sent_action = robot.send_action(init_action)
    # time.sleep(1)


    while True:
        observation = robot.get_observation()
        cam_obs = observation.get(cam_name, None)
        cam_img = None
        if cam_obs is not None:
            if hasattr(cam_obs, "rgb"):
                cam_img = cam_obs.rgb
            elif isinstance(cam_obs, np.ndarray):
                cam_img = cam_obs
            elif hasattr(cam_obs, "image"):
                cam_img = cam_obs.image
        # hwc to chw
        # cam_img = np.ascontiguousarray(cam_img.transpose(2, 0, 1))
        # print(cam_img.shape)
        # breakpoint()
        pred_action = nora.inference(
                    image=cam_img,
                    instruction=args.instruction,
                    unnorm_key=None,
                    unnormalizer=unnormalize,
                )   
        
        pa = np.asarray(pred_action).reshape(-1).astype(float)
        # Nora 输出 [x, y, z, r, p, yaw] —— 这里只取前三个位置量当 Δx,Δy,Δz
        # gripper 目前 Nora 没输出，先置为 1.0（中性，不开不关）。你也可以接键盘事件动态改。
        
        ee_action = {
            "x": float(pa[0]) if pa.size >= 1 else 0.0,
            "y": float(pa[1]) if pa.size >= 2 else 0.0,
            "z": float(pa[2]) if pa.size >= 3 else 0.0,
            "rotation": float(pa[3]) if pa.size >= 4 else 0.0,
            "pitch": float(pa[4]) if pa.size >= 5 else 0.0,
            "yaw": float(pa[5]) if pa.size >= 6 else 0.0,
            "gripper": float(pa[6]),
        }
        print("Model output:", ee_action)
        
        # here check the action shape
        sent_action = robot.send_action(ee_action)
        time.sleep(0.3)
        # breakpoint()
        
    # Disconnect & (optional) push
    robot.disconnect()
    if args.__dict__["dataset.push_to_hub"]:
        ds.push_to_hub(tags=None, private=args.__dict__["dataset.private"])

    if listener is not None:
        listener.stop()


if __name__ == "__main__":
    main()
