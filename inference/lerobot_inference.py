# nora_lerobot_infer.py
import argparse
import numpy as np
import torch
from PIL import Image
from typing import Optional

# ===== Nora（保持你给的实现，略做小修：导入路径与PIL类型名）=====
import PIL.Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download
import json

def normalize_gripper_action(action, binarize=True):
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action

def invert_gripper_action(action):
    action[..., -1] = action[..., -1] * -1.0
    return action

class Nora:
    _ACTION_TOKEN_MIN = 151665
    _ACTION_TOKEN_MAX = 153712

    def __init__(
        self,
        model_path: str = "declare-lab/nora",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
        else:
            self.device = device
            print(f"Using specified device: {self.device}")

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA is not available, but device '{self.device}' was specified.")
            gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
            if gpu_id >= torch.cuda.device_count():
                raise RuntimeError(f"CUDA device {gpu_id} not available. Only {torch.cuda.device_count()} devices found.")

        # Fast tokenizer
        from transformers import AutoProcessor as _AP
        print("Loading fast tokenizer from: physical-intelligence/fast")
        self.fast_tokenizer = _AP.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
        self.fast_tokenizer.action_dim = getattr(self.fast_tokenizer, "action_dim", 7)
        self.fast_tokenizer.time_horizon = getattr(self.fast_tokenizer, "time_horizon", 1)
        print("Setting action_dim to 7 and time_horizon to 1.")

        # Main processor
        print(f"Loading main processor from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Model
        print(f"Loading model from: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            # attn_implementation="flash_attention_2",  # 可按需打开
        ).to(self.device)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.do_sample = False
        self.model.eval()

        # norm_stats（OpenVLA风格）
        repo_id = "declare-lab/nora"
        filename = "norm_stats.json"
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        with open(file_path, "r") as f:
            self.norm_stats = json.load(f)

        print("Model and processors loaded successfully.")

    @torch.inference_mode()
    def inference(self, image: np.ndarray | PIL.Image.Image, instruction: str,
                  unnorm_key: str = None, unnormalizer=None) -> np.ndarray:
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # Nora 期望 224x224
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
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        generated_ids = self.model.generate(**inputs)

        # 找动作token区间的第一个token
        mask = (self._ACTION_TOKEN_MIN <= generated_ids[0]) & (generated_ids[0] <= self._ACTION_TOKEN_MAX)
        idxs = torch.where(mask)[0]
        if len(idxs) == 0:
            raise RuntimeError("No action token found in generated ids.")
        # 取整段动作tokens（一般是连续的一段）
        # 这里简化处理：从第一个落在动作区的token开始，直到该段结束
        start = idxs[0].item()
        end = start
        while end + 1 < generated_ids.shape[1] and \
              self._ACTION_TOKEN_MIN <= generated_ids[0, end + 1] <= self._ACTION_TOKEN_MAX:
            end += 1

        action_token_ids = generated_ids[0, start:end + 1]
        # 映射回fast tokenizer的action范围
        mapped = (action_token_ids - self._ACTION_TOKEN_MIN).tolist()
        # fast_tokenizer.decode -> 你自己的fast解码器应返回形如 (1, T, A) 或 (T, A) 的[-1,1]动作
        output_action = self.fast_tokenizer.decode(mapped)  # 需满足你的fast实现

        # 若提供了LeRobot的unnormalizer，则优先用它反归一
        if unnormalizer is not None:
            if isinstance(output_action, np.ndarray):
                act_np = output_action
            else:
                act_np = np.array(output_action)
            out = unnormalizer({'action': act_np})
            return out['action']

        # 否则用norm_stats（OpenVLA风格）的分位数反归一
        action_norm_stats = self.get_action_stats(unnorm_key)
        action_high = np.array(action_norm_stats["q99"])
        action_low = np.array(action_norm_stats["q01"])
        act = np.array(output_action)  # [-1,1]
        unnorm_actions = 0.5 * (act + 1.0) * (action_high - action_low) + action_low
        return np.array(unnorm_actions[0])

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key: Optional[str]):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from: {list(norm_stats.keys())}"
            )
            unnorm_key = next(iter(norm_stats.keys()))
        assert unnorm_key in norm_stats, f"`unnorm_key` not in: {list(norm_stats.keys())}"
        return unnorm_key

    def get_action_stats(self, unnorm_key: Optional[str] = None):
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]


# ===== LeRobot 反归一化（基于数据集统计）=====
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.configs.types import NormalizationMode, PolicyFeature
from lerobot.policies.normalize import Unnormalize

def build_lerobot_unnormalizer(dataset_repo_id: str, lerobot_dataset_root: str) -> Unnormalize:
    """
    使用 LeRobot 数据集统计构造 Unnormalize。
    这里用 MIN_MAX 对 'action' 维度进行反归一（与 BridgeData V2 一致的常见设置）。
    """
    metadata = LeRobotDatasetMetadata(repo_id=dataset_repo_id, root=lerobot_dataset_root)
    stats = metadata.stats
    features = {'action': PolicyFeature(shape=stats['action']['mean'].shape, type='action')}
    norm_map = {'action': NormalizationMode.MIN_MAX}
    return Unnormalize(features=features, norm_map=norm_map, stats=stats)


# ===== CLI / Demo =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--instruction", type=str,  default="declare-lab/nora", required=True, help="自然语言指令")
    parser.add_argument("--model_path", type=str, default="declare-lab/nora")
    parser.add_argument("--device", type=str, default="cuda:0")  # 如 "cuda:0"
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--dataset_repo_id", type=str, default="xuanyuanj/so101_v3",
                        help="用于构造LeRobot反归一的repo id")
    parser.add_argument("--lerobot_dataset_root", type=str, default="/home/luka/nora/lerobot_so101_v3",
                        help="用于构造LeRobot反归一的数据集根目录")
    parser.add_argument("--map_gripper_to_neg1_pos1", action="store_true",
                        help="若环境期望[-1,1]且训练时invert_grippler_action=True，可将[0,1]映射回[-1,1]")
    parser.add_argument("--invert_gripper", action="store_true", help="将夹爪维度取负（部分环境 -1=open, +1=close）")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # 1) Nora
    nora = Nora(model_path=args.model_path, device=args.device, torch_dtype=dtype)

    # 2) LeRobot Unnormalize
    unnormalize = build_lerobot_unnormalizer(args.dataset_repo_id, args.lerobot_dataset_root)

    # 3) 图像 & 指令
    img = Image.open(args.image_path).convert("RGB")
    instruction = args.instruction

    # 4) 推理（7-DoF；用 LeRobot 的 Unnormalize 反归一）
    actions = nora.inference(
        image=img,
        instruction=instruction,
        unnorm_key=None,          # 如果模型训练时多数据集，可指定key
        unnormalizer=unnormalize  # 使用 LeRobot 的反归一
    )

    # 5) 可选：对夹爪维度做映射/取反 验证
    if args.map_gripper_to_neg1_pos1:
        actions = normalize_gripper_action(actions, binarize=True)
    if args.invert_gripper:
        actions = invert_gripper_action(actions)

    print("Predicted action:", actions)
    print("Action shape:", actions.shape)

if __name__ == "__main__":
    main()