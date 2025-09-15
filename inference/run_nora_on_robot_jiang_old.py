#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAFE, SSH-friendly Nora-on-robot runner with blocking HOMING and terminal hotkeys.
- Replaces global keyboard hook with stdin polling (works over SSH/tmux/VSCode Remote)
- Hotkeys: [f]=toggle FREE/torque, [h]=go HOME (blocking), [q]/[ESC]=quit
- While HOMING or FREE, model inference and action streaming are paused
- Preserves your original args and behavior; adds robust safe-exit
"""

import argparse
import json
import time
import sys
import select
import tty
import termios
import numpy as np
import torch

from nora_long_from_dataset import Nora, build_lerobot_unnormalizer
from lerobot.robots.so101_follower.so101_follower_end_effector import (
    SO101FollowerEndEffector,
    SO101FollowerEndEffectorConfig,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots import make_robot_from_config, so101_follower
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig


# -----------------------
# Helpers
# -----------------------

def to_torch_dtype(dtype_str: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]


def _coerce_camera_config(name: str, cam_dict: dict):
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
    if img is None:
        return None
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    while img.ndim > 2 and img.shape[0] == 1:
        img = np.squeeze(img, axis=0)
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    if expected_w and expected_h and img.ndim == 3:
        H, W = expected_h, expected_w
        if img.shape[0] != H or img.shape[1] != W:
            flat = img.reshape(-1)
            if flat.size >= H * W * 3:
                img = flat[: H * W * 3].reshape(H, W, 3)
            elif flat.size >= H * W:
                g = flat[: H * W].reshape(H, W)
                img = np.stack([g] * 3, axis=-1)
    if img.dtype != np.uint8:
        if img.dtype in (np.float16, np.float32, np.float64) and img.size and float(img.max()) <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]
    return img


class StdinKeyPoller:
    """Non-blocking single-key poller that works over SSH/tmux.
    Usage: with StdinKeyPoller() as kp: ch = kp.poll()
    Returns 'esc' for ESC key, lowercased chars for others, or None.
    """

    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        self.enabled = False

    def __enter__(self):
        tty.setcbreak(self.fd)
        self.enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
        self.enabled = False

    def poll(self):
        if not self.enabled:
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                return "esc"
            return ch.lower()
        return None


def build_robot(args):
    if args.__dict__["robot.type"] not in ("so101_follower_end_effector", "so101_ee"):
        raise NotImplementedError("请设置 --robot.type=so101_follower_end_effector（或 so101_ee）")
    raw = json.loads(args.__dict__["robot.cameras"])
    cameras = {name: _coerce_camera_config(name, cfg) for name, cfg in raw.items()}
    cfg = SO101FollowerEndEffectorConfig(
        port=args.__dict__["robot.port"],
        cameras=cameras,
        id=args.__dict__["robot.id"],
        urdf_path=args.__dict__["robot.urdf_path"],
        target_frame_name=args.__dict__["robot.target_frame"],
        end_effector_step_sizes={
            "x": args.__dict__["robot.ee_step_x"],
            "y": args.__dict__["robot.ee_step_y"],
            "z": args.__dict__["robot.ee_step_z"],
        },
        end_effector_bounds={
            "min": np.array(
                [
                    args.__dict__["robot.ee_min_x"],
                    args.__dict__["robot.ee_min_y"],
                    args.__dict__["robot.ee_min_z"],
                ],
                dtype=np.float32,
            ),
            "max": np.array(
                [
                    args.__dict__["robot.ee_max_x"],
                    args.__dict__["robot.ee_max_y"],
                    args.__dict__["robot.ee_max_z"],
                ],
                dtype=np.float32,
            ),
        },
        max_gripper_pos=args.__dict__["robot.max_gripper_pos"],
    )
    return SO101FollowerEndEffector(cfg)


def parse_args():
    p = argparse.ArgumentParser("Nora on real LeRobot (so101_follower) — SAFE")
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
    # Dataset/recording
    p.add_argument("--dataset.repo_id", type=str, default="xuanyuanj/eval_nora_so101_pour_the_tea")
    p.add_argument("--dataset.single_task", type=str, default="so101_pour_the_tea_ee")
    p.add_argument("--dataset.root", type=str, default="/home/luka/Nora_lerobot/so101_pour_the_tea_ee")
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
    p.add_argument("--instruction", type=str, default="pour tea into the cup.")
    # Unnormalize
    p.add_argument("--lerobot_dataset_root", type=str, default="/home/luka/Nora_lerobot/so101_pour_the_tea_ee")
    p.add_argument("--dataset_for_unnorm", type=str, default="xuanyuanj/eval_nora_so101_pour_the_tea")
    # Camera for Nora
    p.add_argument("--camera_for_nora1", type=str, default="front")
    p.add_argument("--camera_for_nora2", type=str, default="wrist")
    # Optional post-process
    p.add_argument("--map_gripper_to_neg1_pos1", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--invert_gripper", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


# -----------------------
# Torque helpers & HOMING
# -----------------------

def _try_disable_torque(rb):
    for name, arg in [
        ("set_torque_enabled", False),
        ("set_torque", False),
        ("enable_torque", False),
        ("torque_enable", False),
        ("set_compliance", True),
        ("set_free_mode", True),
        ("freeze", False),
    ]:
        fn = getattr(rb, name, None)
        if callable(fn):
            try:
                fn(arg)
                return True
            except Exception:
                pass
    bus = getattr(rb, "bus", None)
    if bus:
        for name, arg in [("torque_enable_all", False), ("set_torque_all", False)]:
            fn = getattr(bus, name, None)
            if callable(fn):
                try:
                    fn(arg)
                    return True
                except Exception:
                    pass
    return False


def _try_enable_torque(rb):
    for name, arg in [
        ("set_torque_enabled", True),
        ("set_torque", True),
        ("enable_torque", True),
        ("torque_enable", True),
        ("set_compliance", False),
        ("set_free_mode", False),
        ("freeze", True),
    ]:
        fn = getattr(rb, name, None)
        if callable(fn):
            try:
                fn(arg)
                return True
            except Exception:
                pass
    bus = getattr(rb, "bus", None)
    if bus:
        for name, arg in [("torque_enable_all", True), ("set_torque_all", True)]:
            fn = getattr(bus, name, None)
            if callable(fn):
                try:
                    fn(arg)
                    return True
                except Exception:
                    pass
    return False


def go_home_blocking(robot, init_action, repeats=12, interval=0.2, settle_s=0.8):
    """Send HOME repeatedly and wait a short time to settle.
    This pauses normal control; use together with op_mode = 'HOMING'.
    """
    try:
        _try_enable_torque(robot)
    except Exception:
        pass

    last_err = None
    for _ in range(repeats):
        try:
            robot.send_action(init_action)
        except Exception as e:
            last_err = e
            print(f"[home] send_action failed once: {e}")
        time.sleep(interval)
    time.sleep(settle_s)
    if last_err:
        print("[home] finished with warnings; near HOME")
    else:
        print("[home] reached HOME (blocking)")


# -----------------------
# Main
# -----------------------

def main():
    args = parse_args()
    dtype = to_torch_dtype(args.dtype)
    print("[Nora] Loading model:", args.model_path)
    nora = Nora(model_path=args.model_path, device=args.device, torch_dtype=dtype)

    unnormalize = build_lerobot_unnormalizer(args.dataset_for_unnorm, args.lerobot_dataset_root)
    robot = build_robot(args)

    if args.display_data:
        _init_rerun(session_name="nora_recording")

    robot.connect()

    # Cameras used for Nora
    cam_name1 = args.camera_for_nora1
    cam_name2 = args.camera_for_nora2
    expected_w1 = robot.cameras[cam_name1].width
    expected_h1 = robot.cameras[cam_name1].height
    expected_w2 = robot.cameras[cam_name2].width
    expected_h2 = robot.cameras[cam_name2].height

    # HOME pose (same as your original)
    init_action = {
        "x": 0.1487499475479126,
        "y": 0.001560,
        "z": 0.00,
        "rotation": 3.1415903,
        "pitch": 0.4194974899291992,
        "yaw": -3.14146447,
        "gripper": 0.02276399,
    }

    # Move to home at start (best-effort)
    try:
        robot.send_action(init_action)
        time.sleep(0.8)
        print("[INIT] moved to home")
    except Exception as e:
        print(f"[INIT] home failed: {e}")

    # Control modes
    free_mode = False
    op_mode = "NORMAL"  # NORMAL / HOMING / FREE
    last_home_ts = 0.0
    HOME_COOLDOWN = 2.0

    # Set up stdin key poller
    with StdinKeyPoller() as keypoll:
        print("[keys] stdin poller active. Press: [f]=toggle torque, [h]=home, [q]/[ESC]=quit")
        try:
            while True:
                # -------- hotkeys --------
                key = keypoll.poll()
                if key in ("f", "h", "q", "esc"):
                    if key == "f":
                        if free_mode:
                            _try_enable_torque(robot)
                            free_mode = False
                            op_mode = "NORMAL"
                            print("[toggle] torque ON")
                        else:
                            _try_disable_torque(robot)
                            free_mode = True
                            op_mode = "FREE"
                            print("[toggle] FREE mode")

                    elif key == "h":
                        now = time.time()
                        if now - last_home_ts >= HOME_COOLDOWN:
                            last_home_ts = now
                            print("[hotkey] go HOME (blocking)")
                            op_mode = "HOMING"
                            try:
                                go_home_blocking(robot, init_action, repeats=12, interval=0.2, settle_s=0.8)
                            except Exception as e:
                                print(f"[hotkey] home failed: {e}")
                            op_mode = "NORMAL"
                        else:
                            print("[hotkey] home ignored (cooldown)")

                    elif key in ("q", "esc"):
                        print("[hotkey] quit requested")
                        break

                # Pause control during HOMING/FREE
                if op_mode in ("HOMING", "FREE"):
                    time.sleep(0.05)
                    continue

                # -------- observation & inference --------
                observation = robot.get_observation()
                cam_obs1 = observation.get(cam_name1, None)
                cam_obs2 = observation.get(cam_name2, None)
                cam_img1 = None
                cam_img2 = None
                if cam_obs1 is not None:
                    if hasattr(cam_obs1, "rgb"):
                        cam_img1 = cam_obs1.rgb
                    elif isinstance(cam_obs1, np.ndarray):
                        cam_img1 = cam_obs1
                    elif hasattr(cam_obs1, "image"):
                        cam_img1 = cam_obs1.image

                if cam_obs2 is not None:
                    if hasattr(cam_obs2, "rgb"):
                        cam_img2 = cam_obs2.rgb
                    elif isinstance(cam_obs2, np.ndarray):
                        cam_img2 = cam_obs2
                    elif hasattr(cam_obs2, "image"):
                        cam_img2 = cam_obs2.image

                cam_img1 = _to_hwc_uint8(cam_img1, expected_w1, expected_h1)
                if cam_img1 is None:
                    time.sleep(0.05)
                    continue
                cam_img2 = _to_hwc_uint8(cam_img2, expected_w2, expected_h2)
                if cam_img2 is None:
                    time.sleep(0.05)
                    continue

                print(f"[Nora Input] img1.after resize: {getattr(cam_img1, 'size', None)}, img2.size: {getattr(cam_img2, 'size', None)}, instruction: {args.instruction}")

                pred_action = nora.inference(
                    image1=cam_img1,
                    image2=cam_img2,
                    instruction=args.instruction,
                    unnorm_key=None,
                    unnormalizer=unnormalize,
                )
                pa = np.asarray(pred_action).reshape(-1).astype(float)
                ee_action = {
                    "x": float(pa[0]) if pa.size >= 1 else 0.0,
                    "y": float(pa[1]) if pa.size >= 2 else 0.0,
                    "z": float(pa[2]) if pa.size >= 3 else 0.0,
                    "rotation": float(pa[3]) if pa.size >= 4 else 0.0,
                    "pitch": float(pa[4]) if pa.size >= 5 else 0.0,
                    "yaw": float(pa[5]) if pa.size >= 6 else 0.0,
                    "gripper": float(pa[6]) if pa.size >= 7 else 0.0,
                }
                print("Model output:", ee_action)

                robot.send_action(ee_action)
                time.sleep(0.3)

        except KeyboardInterrupt:
            print("\n[SAFE-EXIT] Ctrl-C captured, returning home & disabling torque...")

        finally:
            try:
                robot.send_action(init_action)
                time.sleep(0.8)
                print("[SAFE-EXIT] moved to home")
            except Exception as e:
                print(f"[SAFE-EXIT] home failed: {e}")
            try:
                ok = _try_disable_torque(robot)
                print("[SAFE-EXIT] free mode:", ok)
            except Exception as e:
                print(f"[SAFE-EXIT] disable torque failed: {e}")
            try:
                robot.disconnect()
                print("[SAFE-EXIT] disconnected")
            except Exception as e:
                print(f"[SAFE-EXIT] disconnect failed: {e}")


if __name__ == "__main__":
    main()
