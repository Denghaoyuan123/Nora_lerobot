#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
from typing import Dict, Optional, Any, List

import numpy as np
import torch
from PIL import Image
import PIL.Image

# YAML 为首选解析；没装也能回退 JSON
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

# ===== Nora / Qwen-VL =====
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from qwen_vl_utils import process_vision_info

# ===== LeRobot =====
from lerobot.robots import make_robot_from_config
from lerobot.robots import so101_follower  # 如需 so100 可再加
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.utils.robot_utils import busy_wait

# 可选：用于按数据集统计做反归一化（强烈建议）
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.types import NormalizationMode, PolicyFeature
from lerobot.policies.normalize import Unnormalize


# ----------------------------
# Nora helper
# ----------------------------
class Nora:
    _ACTION_TOKEN_MIN = 151665
    _ACTION_TOKEN_MAX = 153712

    def __init__(self, model_path: str, device: str = "cuda:0", torch_dtype: str = "bfloat16"):
        self.device = device
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.dtype = dtype_map[torch_dtype]

        print(f"[Nora] Loading processor: {model_path}")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True
        )

        print(f"[Nora] Loading model: {model_path} (dtype={torch_dtype}, device={device})")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self.dtype
        ).to(self.device)

        # 生成配置：确定性 + 限步
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        # generate() 统一参数
        self.gen_args = dict(
            max_new_tokens=32,   # 关键：每帧最多生成 32 个 token（必要时可降到 24）
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(self.processor.tokenizer, "eos_token_id", None),
        )

        # 预热一次，避免首帧卡顿
        with torch.inference_mode():
            dummy = PIL.Image.new("RGB", (224, 224), (0, 0, 0))
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": dummy, "resized_height": 224, "resized_width": 224},
                {"type": "text", "text": "warmup"},
            ]}]
            text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            imgs, vids = process_vision_info(msgs)
            ins = self.processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt")
            ins = {k: v.to(self.device) for k, v in ins.items()}
            _ = self.model.generate(**ins, **self.gen_args)
        print("[Nora] Warmup done. Ready.")

    @torch.inference_mode()
    def predict_action_tokens(self, image: PIL.Image.Image, instruction: str) -> np.ndarray:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image, "resized_height": 224, "resized_width": 224},
                {"type": "text", "text": instruction},
            ],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs, **self.gen_args)[0]

        mask = (self._ACTION_TOKEN_MIN <= generated_ids) & (generated_ids <= self._ACTION_TOKEN_MAX)
        idxs = torch.where(mask)[0].tolist()
        if not idxs:
            print("[Nora] No action tokens found; returning zeros.")
            return np.zeros(7, dtype=np.float32)

        token_range = self._ACTION_TOKEN_MAX - self._ACTION_TOKEN_MIN + 1
        vals = []
        for i in idxs[:7]:
            v = (generated_ids[i].item() - self._ACTION_TOKEN_MIN) / (token_range - 1) * 2.0 - 1.0
            vals.append(v)
        if len(vals) < 7:
            vals += [0.0] * (7 - len(vals))
        return np.array(vals, dtype=np.float32)

    def __call__(self, image: PIL.Image.Image, instruction: str) -> np.ndarray:
        return self.predict_action_tokens(image, instruction)


# ----------------------------
# Unnormalize builder (optional)
# ----------------------------
def build_unnormalizer(dataset_repo_id: str, dataset_root: str) -> Optional[Unnormalize]:
    if not dataset_repo_id or not dataset_root:
        return None
    try:
        meta = LeRobotDatasetMetadata(repo_id=dataset_repo_id, root=dataset_root)
        stats = meta.stats
        features = {"action": PolicyFeature(shape=stats["action"]["mean"].shape, type="action")}
        norm_map = {"action": NormalizationMode.MIN_MAX}
        print(f"[Unnormalize] Loaded stats from {dataset_repo_id} at {dataset_root}")
        return Unnormalize(features=features, norm_map=norm_map, stats=stats)
    except Exception as e:
        print(f"[Unnormalize] Warning: cannot build unnormalizer: {e}")
        return None


# ----------------------------
# Utils
# ----------------------------
def normalize_gripper_to_sign(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    if out.shape[-1] >= 1:
        out[-1] = np.sign(out[-1])
    return out

def invert_gripper(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    if out.shape[-1] >= 1:
        out[-1] = -out[-1]
    return out

def find_first_image(observation: Dict[str, Any]) -> Optional[PIL.Image.Image]:
    candidates: List[Any] = []

    def scan(v):
        if isinstance(v, dict):
            for vv in v.values():
                scan(vv)
        else:
            candidates.append(v)

    scan(observation)

    for v in candidates:
        try:
            arr = None
            if isinstance(v, np.ndarray) and v.ndim == 3:
                if v.shape[0] in (1, 3):  # C,H,W
                    arr = np.transpose(v, (1, 2, 0))
                elif v.shape[-1] in (1, 3):  # H,W,C
                    arr = v
            if "torch" in str(type(v)):  # torch tensor fallback
                t = v
                if t.ndim == 3:
                    if t.shape[0] in (1, 3):
                        arr = t.detach().cpu().numpy().transpose(1, 2, 0)
                    elif t.shape[-1] in (1, 3):
                        arr = t.detach().cpu().numpy()

            if arr is not None:
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return Image.fromarray(arr)
        except Exception:
            continue
    return None

def map_action_to_robot_keys(pred, action_keys) -> Dict[str, float]:
    # 1) keys：兼容 dict / list / tuple
    if isinstance(action_keys, dict):
        keys = list(action_keys.keys())
    elif isinstance(action_keys, (list, tuple)):
        keys = list(action_keys)
    else:
        keys = list(action_keys)

    pred = np.asarray(pred).reshape(-1)  # 2) 拉平
    dims = min(len(keys), pred.shape[-1])
    out = {k: float(pred[i]) for i, k in enumerate(keys[:dims])}
    for k in keys[dims:]:
        out[k] = 0.0
    return out

def parse_cameras_arg(cams_arg: str) -> Dict[str, dict]:
    if not cams_arg:
        return {}
    if yaml is not None:
        try:
            data = yaml.safe_load(cams_arg)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"[WARN] YAML parse failed: {e}; fallback to JSON.")
    try:
        return json.loads(cams_arg)
    except Exception as e:
        print(f"[WARN] Failed to parse --robot.cameras as JSON: {e}; using empty cameras.")
        return {}

def to_camera_config_dict(cam_cfg: Dict[str, dict]) -> Dict[str, object]:
    cam_objs = {}
    for name, cfg in cam_cfg.items():
        if not isinstance(cfg, dict) or "type" not in cfg:
            raise ValueError(f"Camera '{name}' must be a dict with a 'type' field, got: {cfg}")
        cam_type = str(cfg["type"]).lower().strip()

        if cam_type == "opencv":
            idx = cfg.get("index_or_path", cfg.get("camera_index", 0))
            try:
                if isinstance(idx, str) and idx.isdigit():
                    idx = int(idx)
            except Exception:
                pass
            cam_objs[str(name)] = OpenCVCameraConfig(
                index_or_path=idx,
                width=int(cfg.get("width", 640)),
                height=int(cfg.get("height", 480)),
                fps=int(cfg.get("fps", 30)),
            )
        elif cam_type == "realsense":
            cam_objs[str(name)] = RealSenseCameraConfig(
                serial_number_or_name=str(cfg.get("serial_number_or_name", "")),
                width=int(cfg.get("width", 640)),
                height=int(cfg.get("height", 480)),
                fps=int(cfg.get("fps", 30)),
            )
        else:
            raise ValueError(f"Unsupported camera type '{cam_type}' for camera '{name}'")
    return cam_objs


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Run Nora policy on LeRobot real robot.")
    # Robot
    ap.add_argument("--robot.type", type=str, default="so101_follower")
    ap.add_argument("--robot.port", type=str, default="/dev/ttyACM10")
    ap.add_argument("--robot.id", type=str, default="nora_follower")
    ap.add_argument("--robot.cameras", type=str, default='{}',
                    help="YAML/JSON cameras dict, e.g. {wrist: {type: opencv, index_or_path: /dev/video10, width: 424, height: 240, fps: 15}}")

    # Control vs Policy 频率解耦
    ap.add_argument("--fps", type=int, default=15, help="保持兼容（默认也作为 control_hz）")
    ap.add_argument("--control_hz", type=int, default=None, help="控制环发送频率；默认等于 --fps")
    ap.add_argument("--policy_hz", type=float, default=2.0, help="策略推理频率（Nora生成频率）")
    ap.add_argument("--max_runtime_s", type=float, default=0.0, help="0 => run until Ctrl+C")
    ap.add_argument("--instruction", type=str, default="pick the cube.")

    # Nora
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, choices=["bfloat16", "float16", "float32"], default="bfloat16")

    # Post-process
    ap.add_argument("--map_gripper_to_neg1_pos1", action="store_true")
    ap.add_argument("--invert_gripper", action="store_true")
    ap.add_argument("--clip_action", type=float, default=1.0)

    # Optional unnormalize
    ap.add_argument("--dataset.repo_id", type=str, default="")
    ap.add_argument("--dataset.root", type=str, default="")

    args = ap.parse_args()

    # 相机配置
    cam_cfg_dict = parse_cameras_arg(args.__dict__["robot.cameras"])
    try:
        camera_objs = to_camera_config_dict(cam_cfg_dict)
    except Exception as e:
        print(f"[WARN] camera config error: {e}; continuing without cameras.")
        camera_objs = {}

    # 机器人配置（so101）
    if args.__dict__["robot.type"] != "so101_follower":
        raise ValueError(f"Unsupported --robot.type: {args.__dict__['robot.type']} (only so101_follower here)")
    robot_cfg = so101_follower.SO101FollowerConfig(
        port=args.__dict__["robot.port"], id=args.__dict__["robot.id"], cameras=camera_objs
    )
    robot = make_robot_from_config(robot_cfg)
    print(f"[Robot] Created robot: {robot.name} ({args.__dict__['robot.type']})")
    print(f"[Robot] action_features = {robot.action_features}")

    robot.connect()
    print("[Robot] Connected.")

    # Unnormalize
    unnormalize = build_unnormalizer(args.__dict__["dataset.repo_id"], args.__dict__["dataset.root"])

    # Nora
    nora = Nora(model_path=args.model_path, device=args.device, torch_dtype=args.dtype)

    # 频率参数
    control_hz = args.control_hz or args.fps
    policy_hz = max(0.5, float(args.policy_hz))
    control_period = 1.0 / control_hz
    policy_period = 1.0 / policy_hz

    print(f"[Freq] control_hz={control_hz}, policy_hz={policy_hz}")

    t_start = time.perf_counter()
    last_policy_t = 0.0
    last_action_dict: Optional[Dict[str, float]] = None

    try:
        print("[Loop] Starting control loop. Ctrl+C to stop.")
        while True:
            loop_start = time.perf_counter()
            if args.max_runtime_s > 0 and (loop_start - t_start) >= args.max_runtime_s:
                print("[Loop] Reached max runtime. Stopping.")
                break

            # 1) 取观测
            obs = robot.get_observation()

            # 2) 拿一张图
            img = find_first_image(obs)
            if img is None:
                print("[Loop] No image found; skip this frame.")
                busy_wait(max(0.0, control_period - (time.perf_counter() - loop_start)))
                continue

            now = loop_start
            need_policy = (now - last_policy_t) >= policy_period

            # 3) 仅按 policy_hz 执行 Nora 推理，其余帧重发 last_action
            if need_policy:
                t0 = time.perf_counter()
                pred = None
                try:
                    pred = nora(img, args.instruction)
                except Exception as e:
                    print(f"[Warn] Nora inference failed: {e}")
                dt = time.perf_counter() - t0
                if dt > 0.25:
                    print(f"[Perf] generate dt={dt*1000:.1f} ms")

                if pred is not None:
                    # 4) 可选反归一化 + 后处理
                    try:
                        if unnormalize is not None:
                            if pred.ndim > 1:
                                pred = pred.reshape(-1)
                            expected = unnormalize.features["action"].shape[-1]
                            if pred.shape[-1] < expected:
                                pred = np.pad(pred, (0, expected - pred.shape[-1]), mode="constant", constant_values=0.0)
                            elif pred.shape[-1] > expected:
                                pred = pred[:expected]
                            tens = torch.from_numpy(pred).float()
                            out = unnormalize({"action": tens})
                            pred = out["action"].detach().cpu().numpy().reshape(-1)
                    except Exception as e:
                        print(f"[Unnormalize] failed: {e}. Continue with raw pred.")

                    if args.map_gripper_to_neg1_pos1:
                        pred = normalize_gripper_to_sign(pred)
                    if args.invert_gripper:
                        pred = invert_gripper(pred)
                    if args.clip_action > 0:
                        pred = np.clip(pred, -args.clip_action, args.clip_action)

                    last_action_dict = map_action_to_robot_keys(pred, list(robot.action_features.keys()))
                    last_policy_t = now
                    # 可选调试：打印前三个值
                    # print(f"[Action] {list(last_action_dict.items())[:3]} ...")

            # 5) 若本帧没有新动作，就重发上一次（保持/心跳）
            action_to_send = last_action_dict
            if action_to_send is None:
                # 初期可能还没推理到第一帧，先空转等待
                busy_wait(max(0.0, control_period - (time.perf_counter() - loop_start)))
                continue

            _sent = robot.send_action(action_to_send)

            # 6) 控制频率节流
            busy_wait(max(0.0, control_period - (time.perf_counter() - loop_start)))

    except KeyboardInterrupt:
        print("\n[Loop] Interrupted by user (Ctrl+C).")
    finally:
        robot.disconnect()
        print("[Robot] Disconnected.")


if __name__ == "__main__":
    main()
