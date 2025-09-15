import argparse
import torch
from nora import Nora, normalize_gripper_action, invert_gripper_action
from lerobot_inference import build_lerobot_unnormalizer
from lerobot.robots.so101_follower import SO101Follower   # 用 so101_follower

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/home/luka/Nora_lerobot/lerobot_training/hf_model_40000",
                        help="训练好的 Nora 模型路径或 HuggingFace repo id")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--dataset_repo_id", type=str, default="xuanyuanj/so101_v3",
                        help="用于构造 LeRobot Unnormalize 的数据集 repo id")
    parser.add_argument("--lerobot_dataset_root", type=str, default="/home/luka/nora/lerobot_so101_v3",
                        help="用于构造 LeRobot Unnormalize 的数据集根目录")
    parser.add_argument("--instruction", type=str, default="pick up the red cube")
    parser.add_argument("--camera_keys", nargs="+", default=["front"],
                        help="obs['images'] 中使用的相机 key，可以多个，例如: --camera_keys front wrist")
    parser.add_argument("--map_gripper_to_neg1_pos1", action="store_true",
                        help="若机器人期望 [-1,1]，则把 [0,1] 的 gripper 映射回 [-1,1]")
    parser.add_argument("--invert_gripper", action="store_true",
                        help="是否对 gripper 取反（部分机器人 -1=open, +1=close）")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0",
                        help="机器人串口/通信端口")
    args = parser.parse_args()

    # ===== Dtype 映射 =====
    dtype_map = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # ===== 初始化 Nora =====
    nora = Nora(model_path=args.model_path, device=args.device, torch_dtype=dtype)

    # ===== 构造 LeRobot Unnormalize =====
    unnormalize = build_lerobot_unnormalizer(args.dataset_repo_id, args.lerobot_dataset_root)

    # ===== 初始化 so101_follower =====
    robot = SO101Follower(port=args.port, use_camera=True)

    print("=== Nora x LeRobot (so101_follower) started ===")
    print(f"Instruction: {args.instruction}")
    print(f"Camera keys: {args.camera_keys}")

    while True:
        # 1) 获取观测
        obs = robot.get_obs()
        imgs = []
        for key in args.camera_keys:
            if key not in obs["images"]:
                raise KeyError(f"相机 {key} 不在 obs['images'] 中，实际 keys: {list(obs['images'].keys())}")
            imgs.append(obs["images"][key])

        # 2) Nora 推理（当前只用第一路相机）
        act = nora.inference(
            image=imgs[0],
            instruction=args.instruction,
            unnormalizer=unnormalize
        )

        # 3) 夹爪映射处理
        if args.map_gripper_to_neg1_pos1:
            act = normalize_gripper_action(act, binarize=True)
        if args.invert_gripper:
            act = invert_gripper_action(act)

        # 4) 执行动作
        robot.step(act)

        print("Predicted action:", act)

if __name__ == "__main__":
    main()
