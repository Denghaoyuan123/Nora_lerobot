#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test the minimal bridge functions in nora_function.py:
# 1) get_joint_positions(robot)
# 2) get_front_image(robot)
# 3) send_joint(robot, action)
#
# Usage examples:
#   python test_nora_functions.py --port /dev/ttyACM5 --mode joints
#   python test_nora_functions.py --port /dev/ttyACM5 --mode image --save front.jpg
#   python test_nora_functions.py --port /dev/ttyACM5 --mode step_zero --enable-move
#   python test_nora_functions.py --port /dev/ttyACM5 --mode step_nudge --enable-move --duration 2.0
#
# This script will NOT move the robot unless you pass --enable-move.

import argparse
import time
import sys
from datetime import datetime

import numpy as np

# Import user's functions
from nora_function import (
    create_so101_robot,
    get_joint_positions,
    get_front_image,
    send_joint,
    get_action_names,
)


def test_joints(robot, n=5, interval=0.2):
    """Poll joint positions a few times and print them."""
    for i in range(n):
        q = get_joint_positions(robot)
        print(f"[{i+1}/{n}] qpos shape={q.shape}; values={np.round(q, 4)}")
        time.sleep(interval)


def test_image(robot, save_path=None, show=True):
    """Grab one front image, print shape, optionally show/save."""
    img = get_front_image(robot)
    print(f"Front image shape={img.shape}, dtype={img.dtype}")
    if save_path:
        try:
            from PIL import Image
            Image.fromarray(img).save(save_path)
            print(f"Saved front image to: {save_path}")
        except Exception as e:
            print(f"Failed to save via PIL ({e}), trying imageio...")
            try:
                import imageio.v2 as imageio
                imageio.imwrite(save_path, img)
                print(f"Saved front image to: {save_path}")
            except Exception as e2:
                print(f"Failed to save image: {e2}")

    if show:
        try:
            import cv2
            cv2.imshow("front", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(800)  # show briefly
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"OpenCV display failed ({e}). Skipping window display.")


def test_send_zero(robot, duration=1.0, rate=20):
    """Send all-zero action for a short duration (requires --enable-move)."""
    names = get_action_names(robot)
    print("Action names/order:", names)
    act = np.zeros(len(names), dtype=np.float32)
    steps = max(1, int(duration * rate))
    dt = 1.0 / rate
    print(f"Sending zero action for {duration}s @ {rate} Hz ({steps} steps)." )
    for _ in range(steps):
        send_joint(robot, act)
        time.sleep(dt)


def test_send_nudge(robot, duration=2.0, rate=20, amp=0.1, freq=0.5):
    """Send small sinusoidal nudges on first 1-2 joints (requires --enable-move)."""
    names = get_action_names(robot)
    print("Action names/order:", names)
    steps = max(1, int(duration * rate))
    dt = 1.0 / rate
    t = 0.0
    print(f"Sending small nudges (amp={amp}, freq={freq}Hz) for {duration}s @ {rate} Hz.")
    for _ in range(steps):
        act = np.zeros(len(names), dtype=np.float32)
        if len(names) > 0:
            act[0] = amp * np.sin(2 * np.pi * freq * t)
        if len(names) > 1:
            act[1] = amp * np.sin(2 * np.pi * freq * t + np.pi / 2)
        # keep gripper unchanged (if exists) by default
        send_joint(robot, act)
        time.sleep(dt)
        t += dt


def main():
    parser = argparse.ArgumentParser(description="Tester for nora_function bridge APIs.")
    parser.add_argument("--port", type=str, default="/dev/ttyACM5", help="Robot serial/comm port")
    parser.add_argument(
        "--front_cam",
        type=str,
        default="0",
        help="OpenCV camera index or path for the 'front' camera (e.g., 0, 2, /dev/video2)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["joints", "image", "step_zero", "step_nudge"],
        help="Test mode: read joints / grab image / send zero / small nudge",
    )
    parser.add_argument("--save", type=str, default=None, help="If set, save grabbed image to this path")
    parser.add_argument("--duration", type=float, default=1.0, help="Send duration for step modes (seconds)")
    parser.add_argument("--rate", type=int, default=20, help="Hz for sending actions")
    parser.add_argument("--amp", type=float, default=0.1, help="Amplitude for nudge")
    parser.add_argument("--freq", type=float, default=0.5, help="Frequency (Hz) for nudge")
    parser.add_argument(
        "--enable-move",
        action="store_true",
        help="Required for step modes to actually move the robot (safety gate)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not pop up a window when testing image grabbing",
    )
    args = parser.parse_args()

    # Convert front_cam to int if it's a digit
    front_cam = args.front_cam
    if isinstance(front_cam, str) and front_cam.isdigit():
        front_cam = int(front_cam)

    # Create and connect robot
    robot = create_so101_robot(
        port=args.port,
        front_index_or_path=front_cam,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    try:
        if args.mode == "joints":
            test_joints(robot)
        elif args.mode == "image":
            test_image(robot, save_path=args.save, show=not args.no_show)
        elif args.mode in ("step_zero", "step_nudge"):
            if not args.enable_move:
                print(
                    "Refusing to move the robot without --enable-move. "
                    "Rerun with --enable-move if you are in a safe environment."
                )
                sys.exit(2)
            if args.mode == "step_zero":
                test_send_zero(robot, duration=args.duration, rate=args.rate)
            else:
                test_send_nudge(
                    robot,
                    duration=args.duration,
                    rate=args.rate,
                    amp=args.amp,
                    freq=args.freq,
                )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")
    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
