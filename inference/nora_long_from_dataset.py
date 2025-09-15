# nora_lerobot_infer_enhanced_fixed.py
import argparse
import numpy as np
import torch
from PIL import Image
from typing import Optional, Dict, List, Tuple
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ===== Nora（修复版本）=====
import PIL.Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download
from torchvision.transforms.functional import to_pil_image
import torchvision

def normalize_gripper_action(action, binarize=True):
    """归一化夹爪动作到[-1,1]范围"""
    if len(action.shape) > 1:
        action = action.copy()
    else:
        action = action.copy()
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action

def invert_gripper_action(action):
    """反转夹爪动作符号"""
    if len(action.shape) > 1:
        action = action.copy()
    else:
        action = action.copy()
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

        # Fast tokenizer - 使用更健壮的初始化
        from transformers import AutoProcessor as _AP
        print("Loading fast tokenizer from: physical-intelligence/fast")
        
        self.fast_tokenizer = _AP.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
        self.fast_tokenizer_available = True
        
        self.fast_tokenizer.action_dim = 7 # Set default if not in config
        print("Setting action_dim  to 7.")
           
        self.fast_tokenizer.time_horizon = 5 # Set default if not in config
        print("Setting time horizon to 1.")

        # Main processor
        print(f"Loading main processor from: {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Model
        print(f"Loading model from: {model_path}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.do_sample = False
        self.model.eval()

        # norm_stats（OpenVLA风格）
        try:
            repo_id = "declare-lab/nora"
            filename = "norm_stats.json"
            file_path = hf_hub_download(repo_id=repo_id, filename=filename)
            with open(file_path, "r") as f:
                self.norm_stats = json.load(f)
            print("Loaded norm_stats successfully.")
            # breakpoint()
        except Exception as e:
            print(f"Warning: Could not load norm_stats: {e}")
            # breakpoint()
            self.norm_stats = {}

        print("Model and processors loaded successfully.")

    def _decode_action_tokens_fallback(self, mapped_tokens: List[int]) -> np.ndarray:
        """备用的token解码方法"""
        print(f"Using fallback token decoding for {len(mapped_tokens)} tokens")
        
        # 计算token范围
        token_range = self._ACTION_TOKEN_MAX - self._ACTION_TOKEN_MIN + 1
        
        # 如果tokens不足7个，用零填充
        if len(mapped_tokens) < 7:
            print(f"Padding {len(mapped_tokens)} tokens to 7 dimensions")
            padded_tokens = mapped_tokens + [0] * (7 - len(mapped_tokens))
        else:
            padded_tokens = mapped_tokens[:7]  # 截取前7个
        
        # 将token映射到[-1, 1]范围
        normalized_action = []
        for token in padded_tokens:
            # 线性映射: [0, token_range-1] -> [-1, 1]
            normalized_val = (token / (token_range - 1)) * 2 - 1
            normalized_action.append(normalized_val)
        
        return np.array(normalized_action, dtype=np.float32).reshape(1, 7)

    @torch.inference_mode()
    def inference(self, image1: np.ndarray | PIL.Image.Image,
                  image2: np.ndarray | PIL.Image.Image,  
                  instruction: str,
                  unnorm_key: str = None, unnormalizer=None) -> np.ndarray:
        if not isinstance(image1, PIL.Image.Image):
            image1 = PIL.Image.fromarray(image1)
        if not isinstance(image2, PIL.Image.Image):
            image2 = PIL.Image.fromarray(image2)

        image1 = image1.resize((224, 224))
        image2 = image2.resize((224, 224))
        print(f"[Nora Input] img1.after resize: {getattr(image1, 'size', None)}, img2.size: {getattr(image2, 'size', None)}, instruction: {instruction}")


        # Nora 期望 224x224
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image1, "resized_height": 224, "resized_width": 224,},
                {"type": "image", "image": image2, "resized_height": 224, "resized_width": 224,},
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

        # 找动作token区间
        start_idx = (self._ACTION_TOKEN_MIN <= generated_ids[0]) & (generated_ids[0] <= self._ACTION_TOKEN_MAX)
        start_idx = torch.where(start_idx)[0]

        
        # 取连续的动作tokens
        if len(start_idx) > 0:
            start_index = start_idx[0].item()
        else:
            start_index = None  # or -1 to indicate not found

        # 先decode
        output_action = self.fast_tokenizer.decode([generated_ids[0][start_idx] - self._ACTION_TOKEN_MIN])
        if unnormalizer is not None:
            try:
                act_np = np.array(output_action, dtype=np.float32)
                
                # 确保shape正确
                if act_np.ndim == 2 and act_np.shape[0] == 1:
                    act_np = act_np[0]  # (1, A) -> (A)
                elif act_np.ndim == 3:
                    act_np = act_np[0, 0]  # (1, 1, A) -> (A)
                
                # 处理维度匹配
                expected_action_dim = unnormalizer.features['action'].shape[-1]
                current_action_dim = act_np.shape[-1]
                # breakpoint()
                
                print(f"Action dimensions: expected={expected_action_dim}, got={current_action_dim}")
                
                if current_action_dim > expected_action_dim:
                    act_np = act_np[:expected_action_dim]
                    print(f"Truncated action to {expected_action_dim} dimensions")
                elif current_action_dim < expected_action_dim:
                    pad_size = expected_action_dim - current_action_dim
                    act_np = np.pad(act_np, (0, pad_size), mode='constant', constant_values=0)
                    print(f"Padded action to {expected_action_dim} dimensions")
                
                # 反归一化
                if hasattr(unnormalizer.stats['action']['mean'], 'device'):
                    act_tensor = torch.from_numpy(act_np).float()
                    out = unnormalizer({'action': act_tensor})
                    result = out['action'].cpu().numpy()
                else:
                    out = unnormalizer({'action': act_np})
                    result = out['action']
                    if hasattr(result, 'cpu'):
                        result = result.cpu().numpy()
                    elif hasattr(result, 'numpy'):
                        result = result.numpy()
                
                return result.astype(np.float64)  # 确保返回标准numpy类型
                
            except Exception as e:
                print(f"Unnormalizer failed: {e}, falling back to norm_stats")

        # 使用norm_stats反归一化
        if self.norm_stats:
            try:
                action_norm_stats = self.get_action_stats(unnorm_key)
                action_high = np.array(action_norm_stats["q99"], dtype=np.float64)
                action_low = np.array(action_norm_stats["q01"], dtype=np.float64)
                act = np.array(output_action, dtype=np.float64)
                if act.ndim == 2:
                    act = act[0]
                unnorm_actions = 0.5 * (act + 1.0) * (action_high - action_low) + action_low
                return unnorm_actions
            except Exception as e:
                print(f"norm_stats denormalization failed: {e}")
        
        # 最后的备用方案
        if output_action.ndim == 2:
            return output_action[0].astype(np.float64)
        return output_action.astype(np.float64)

    def get_action_stats(self, unnorm_key: Optional[str] = None):
        if not self.norm_stats:
            raise RuntimeError("norm_stats not available")
            
        if unnorm_key is None and len(self.norm_stats) > 1:
            preferred_keys = ['bridge_orig', 'fractal20220817_data', 'berkeley_autolab_ur5', 'roboturk']
            for key in preferred_keys:
                if key in self.norm_stats:
                    unnorm_key = key
                    print(f"Auto-selected unnorm_key: {unnorm_key}")
                    break
            if unnorm_key is None:
                unnorm_key = next(iter(self.norm_stats.keys()))
                print(f"Using first available unnorm_key: {unnorm_key}")
        
        if unnorm_key is None:
            unnorm_key = next(iter(self.norm_stats.keys()))
        
        if unnorm_key not in self.norm_stats:
            raise ValueError(f"`unnorm_key` not in: {list(self.norm_stats.keys())}")
        
        return self.norm_stats[unnorm_key]["action"]


# ===== LeRobot 反归一化 =====
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.configs.types import NormalizationMode, PolicyFeature
from lerobot.policies.normalize import Unnormalize

def build_lerobot_unnormalizer(dataset_repo_id: str, lerobot_dataset_root: str) -> Unnormalize:
    """构造LeRobot反归一化器"""
    metadata = LeRobotDatasetMetadata(repo_id=dataset_repo_id, root=lerobot_dataset_root)
    stats = metadata.stats
    features = {'action': PolicyFeature(shape=stats['action']['mean'].shape, type='action')}
    norm_map = {'action': NormalizationMode.MIN_MAX}
    return Unnormalize(features=features, norm_map=norm_map, stats=stats)


# ===== 性能评估工具（修复JSON序列化问题）=====
def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

class InferenceEvaluator:
    def __init__(self, dataset_repo_id: str, dataset_root: str, output_dir: str = "./evaluation_results"):
        self.dataset_repo_id = dataset_repo_id
        self.dataset_root = dataset_root
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集
        self.dataset = LeRobotDataset(
            repo_id=dataset_repo_id,
            root=dataset_root,
            video_backend="pyav",
        )
        print(f"Loaded dataset with {len(self.dataset)} samples")
        
        # 存储结果
        self.results = []
        self.inference_times = []
        
    def get_sample(self, idx: int) -> Tuple[PIL.Image.Image, np.ndarray, str]:
        """获取数据集样本"""
        ex = self.dataset[idx]
        img_chw_1 = ex["observation.images.front"]
        img_chw_2 = ex["observation.images.wrist"]
        action = ex["action"].float().numpy()
        lang = ex["task"]
        img_pil_1 = to_pil_image(img_chw_1)
        img_pil_2 = to_pil_image(img_chw_2)
        return img_pil_1, img_pil_2, action, lang

    def compute_metrics(self, predicted: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """计算各种性能指标"""
        # 确保shape一致
        if predicted.ndim == 2 and predicted.shape[0] == 1:
            predicted = predicted[0]
        if ground_truth.ndim == 2 and ground_truth.shape[0] == 1:
            ground_truth = ground_truth[0]
            
        # 确保维度匹配
        min_dim = min(len(predicted), len(ground_truth))
        predicted = predicted[:min_dim]
        ground_truth = ground_truth[:min_dim]
        
        # L2 误差
        l2_error = float(np.linalg.norm(predicted - ground_truth))
        
        # L1 误差  
        l1_error = float(np.mean(np.abs(predicted - ground_truth)))
        
        # 各维度的误差
        per_dim_l1 = np.abs(predicted - ground_truth)
        per_dim_l2 = (predicted - ground_truth) ** 2
        
        # 位置误差（前3维）和方向误差（3-6维）
        if min_dim >= 6:
            pos_l1 = float(np.mean(per_dim_l1[:3]))
            pos_l2 = float(np.sqrt(np.mean(per_dim_l2[:3])))
            rot_l1 = float(np.mean(per_dim_l1[3:6]))
            rot_l2 = float(np.sqrt(np.mean(per_dim_l2[3:6])))
        else:
            pos_l1 = pos_l2 = rot_l1 = rot_l2 = 0.0
            
        # 夹爪误差（最后一维）
        if min_dim >= 7:
            gripper_error = float(per_dim_l1[-1])
        elif min_dim >= 1:
            gripper_error = float(per_dim_l1[-1])
        else:
            gripper_error = 0.0
            
        return {
            'l2_error': l2_error,
            'l1_error': l1_error,
            'position_l1': pos_l1,
            'position_l2': pos_l2,
            'rotation_l1': rot_l1,
            'rotation_l2': rot_l2,
            'gripper_error': gripper_error,
            'per_dim_l1': [float(x) for x in per_dim_l1],
            'per_dim_l2': [float(x) for x in per_dim_l2]
        }
    
    def evaluate_batch(self, nora_model: Nora, unnormalizer, indices: List[int], 
                      map_gripper: bool = False, invert_gripper: bool = False, instruction_input: str = "") -> Dict:
        """批量评估"""
        batch_results = []
        batch_times = []
        
        print(f"Evaluating {len(indices)} samples...")
        
        for idx in tqdm(indices):
            try:
                # 获取样本
                img1, img2, gt_action, instruction = self.get_sample(idx)
                img1 = img1.resize((224, 224))
                img2 = img2.resize((224, 224))

                # 推理计时
                start_time = time.time()
                pred_action = nora_model.inference(
                    image1=img1,
                    image2=img2,
                    instruction=instruction_input,
                    unnorm_key=None,
                    unnormalizer=unnormalizer
                )
                inference_time = time.time() - start_time
                
                # 后处理
                # if map_gripper and len(pred_action) >= 7:
                #     pred_action = normalize_gripper_action(pred_action, binarize=True)
                # if invert_gripper and len(pred_action) >= 7:
                #     pred_action = invert_gripper_action(pred_action)
                
                # 计算指标
                metrics = self.compute_metrics(pred_action, gt_action)
                
                # 记录结果（确保所有类型都是JSON可序列化的）
                result = {
                    'sample_idx': int(idx),
                    'instruction': str(instruction),
                    'inference_time': float(inference_time),
                    'predicted_action': [float(x) for x in pred_action],
                    'ground_truth_action': [float(x) for x in gt_action],
                    **metrics
                }
                
                batch_results.append(result)
                batch_times.append(float(inference_time))
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
                
        return {
            'results': batch_results,
            'times': batch_times
        }
    
    def save_results(self, results: Dict, filename: str = "evaluation_results.json"):
        """保存结果到文件"""
        # 转换numpy类型确保JSON可序列化
        results_clean = convert_numpy_types(results)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        print(f"Results saved to {output_path}")
    
    def generate_report(self, results: Dict) -> Dict:
        """生成评估报告"""
        batch_results = results['results']
        batch_times = results['times']
        
        if not batch_results:
            return {"error": "No valid results to analyze"}
        
        # 统计指标
        metrics_df = pd.DataFrame(batch_results)
        
        report = {
            'summary': {
                'total_samples': len(batch_results),
                'avg_inference_time': float(np.mean(batch_times)),
                'std_inference_time': float(np.std(batch_times)),
                'min_inference_time': float(np.min(batch_times)),
                'max_inference_time': float(np.max(batch_times)),
            },
            'performance_metrics': {
                'avg_l1_error': float(metrics_df['l1_error'].mean()),
                'std_l1_error': float(metrics_df['l1_error'].std()),
                'avg_l2_error': float(metrics_df['l2_error'].mean()),
                'std_l2_error': float(metrics_df['l2_error'].std()),
                'avg_position_l1': float(metrics_df['position_l1'].mean()),
                'avg_position_l2': float(metrics_df['position_l2'].mean()),
                'avg_rotation_l1': float(metrics_df['rotation_l1'].mean()),
                'avg_rotation_l2': float(metrics_df['rotation_l2'].mean()),
                'avg_gripper_error': float(metrics_df['gripper_error'].mean()),
            }
        }
        
        return report
    
    def plot_results(self, results: Dict):
        """绘制结果图表"""
        batch_results = results['results']
        if not batch_results:
            print("No results to plot")
            return
            
        # 准备数据
        metrics_df = pd.DataFrame(batch_results)
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Nora Inference Performance Analysis', fontsize=16)
        
        # 1. 推理时间分布
        axes[0, 0].hist(metrics_df['inference_time'], bins=20, alpha=0.7)
        axes[0, 0].set_title('Inference Time Distribution')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Count')
        
        # 2. L1/L2误差分布
        axes[0, 1].hist(metrics_df['l1_error'], bins=20, alpha=0.7, label='L1 Error')
        axes[0, 1].hist(metrics_df['l2_error'], bins=20, alpha=0.7, label='L2 Error')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].set_xlabel('Error')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend()
        
        # 3. 位置vs旋转误差
        axes[0, 2].scatter(metrics_df['position_l1'], metrics_df['rotation_l1'], alpha=0.6)
        axes[0, 2].set_title('Position vs Rotation Error')
        axes[0, 2].set_xlabel('Position L1 Error')
        axes[0, 2].set_ylabel('Rotation L1 Error')
        
        # 4. 各维度误差热图
        per_dim_errors = np.array([r['per_dim_l1'] for r in batch_results])
        if per_dim_errors.size > 0:
            im = axes[1, 0].imshow(per_dim_errors.T, cmap='viridis', aspect='auto')
            axes[1, 0].set_title('Per-dimension L1 Errors')
            axes[1, 0].set_xlabel('Sample Index')
            axes[1, 0].set_ylabel('Action Dimension')
            plt.colorbar(im, ax=axes[1, 0])
        
        # 5. 误差随样本变化
        axes[1, 1].plot(metrics_df['l1_error'], label='L1 Error', alpha=0.7)
        axes[1, 1].plot(metrics_df['l2_error'], label='L2 Error', alpha=0.7)
        axes[1, 1].set_title('Error Over Samples')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].legend()
        
        # 6. 夹爪误差分布
        if 'gripper_error' in metrics_df.columns:
            axes[1, 2].hist(metrics_df['gripper_error'], bins=20, alpha=0.7)
            axes[1, 2].set_title('Gripper Error Distribution')
            axes[1, 2].set_xlabel('Gripper Error')
            axes[1, 2].set_ylabel('Count')
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = self.output_dir / "evaluation_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Plots saved to {plot_path}")


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser(description="Enhanced Nora LeRobot Inference Testing (Fixed)")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, 
                        default="/home/luka/Nora_lerobot/lerobot_training/long_hf_model/lerobot_training/nora_finetune_pick_up_long_0830/hf_model_40000")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    
    # 数据集参数
    parser.add_argument("--dataset_repo_id", type=str, default="xuanyuanj/so101_v3")
    parser.add_argument("--instruction", type=str, default="pour tea into the cup.")
    parser.add_argument("--lerobot_dataset_root", type=str, default="/home/luka/Nora_lerobot/so101_pour_the_tea_ee")
    
    # 评估参数
    parser.add_argument("--num_samples", type=int, default=30, help="要测试的样本数量")
    parser.add_argument("--start_idx", type=int, default=0, help="起始样本索引")
    parser.add_argument("--random_sampling", action="store_true", help="随机采样而非连续采样")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="结果输出目录")
    parser.add_argument("--unnorm_key", type=str, default=None, help="用于反归一的数据集key")
    
    # 后处理参数
    parser.add_argument("--map_gripper_to_neg1_pos1", action="store_true")
    parser.add_argument("--invert_gripper", action="store_true")
    
    # 运行模式
    parser.add_argument("--mode", type=str, default="single", 
                        choices=["single", "evaluate", "analyze"], 
                        help="运行模式：single=单个样本，evaluate=批量评估，analyze=分析已有结果")
    parser.add_argument("--results_file", type=str, default="evaluation_results.json", 
                        help="用于分析模式的结果文件")
    
    args = parser.parse_args()
    
    if args.mode == "analyze":
        # 分析已有结果
        evaluator = InferenceEvaluator(args.dataset_repo_id, args.lerobot_dataset_root, args.output_dir)
        results_path = Path(args.output_dir) / args.results_file
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            report = evaluator.generate_report(results)
            print("\n=== Evaluation Report ===")
            print(json.dumps(report, indent=2))
            evaluator.plot_results(results)
        else:
            print(f"Results file not found: {results_path}")
        return
    
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # 初始化模型和工具
    print("=== Initializing Nora Model ===")
    print("Load model", args.model_path)
    nora = Nora(model_path=args.model_path, device=args.device, torch_dtype=dtype)
    
    print("=== Building LeRobot Unnormalizer ===")
    unnormalize = build_lerobot_unnormalizer(args.dataset_repo_id, args.lerobot_dataset_root)
    
    print("=== Setting up Evaluator ===")
    evaluator = InferenceEvaluator(args.dataset_repo_id, args.lerobot_dataset_root, args.output_dir)
    
    if args.mode == "single":
        # 单个样本测试
        print("=== Single Sample Test ===")
        img1, img2, gt_action, instruction = evaluator.get_sample(args.start_idx)
        print(f"Instruction: {args.instruction}")
        print(f"Ground truth action shape: {gt_action.shape}")
        print(f"Ground truth action: {gt_action}")
        print(f"[Nora Input] img1.size: {getattr(img1, 'size', None)}, img2.size: {getattr(img2, 'size', None)}, instruction: {args.instruction}")
        # breakpoint()
        
        start_time = time.time()
        pred_action = nora.inference(
            image1=img1,
            image2=img2,
            instruction=args.instruction,
            unnorm_key=args.unnorm_key,
            unnormalizer=unnormalize
        )
        inference_time = time.time() - start_time
        
        # print(f"Predicted action shape: {pred_action.shape}")
        print(f"Predicted action (raw): {pred_action}")
        # breakpoint()
        if args.map_gripper_to_neg1_pos1 and len(pred_action) >= 7:
            pred_action = normalize_gripper_action(pred_action, binarize=True)
            print(f"Predicted action (gripper normalized): {pred_action}")
        if args.invert_gripper and len(pred_action) >= 7:
            pred_action = invert_gripper_action(pred_action)
            print(f"Predicted action (gripper inverted): {pred_action}")
            
        metrics = evaluator.compute_metrics(pred_action, gt_action)
        
        print(f"\nResults:")
        print(f"Inference time: {inference_time:.4f}s")
        print(f"L1 Error: {metrics['l1_error']:.4f}")
        print(f"L2 Error: {metrics['l2_error']:.4f}")
        print(f"Position L1: {metrics['position_l1']:.4f}")
        print(f"Rotation L1: {metrics['rotation_l1']:.4f}")
        print(f"Gripper Error: {metrics['gripper_error']:.4f}")
        
    elif args.mode == "evaluate":
        # 批量评估
        print("=== Batch Evaluation ===")
        
        # 确定要测试的样本索引
        if args.random_sampling:
            indices = np.random.choice(len(evaluator.dataset), 
                                     min(args.num_samples, len(evaluator.dataset)), 
                                     replace=False).tolist()
        else:
            end_idx = min(args.start_idx + args.num_samples, len(evaluator.dataset))
            indices = list(range(args.start_idx, end_idx))
        
        print(f"Testing {len(indices)} samples")
        if len(indices) <= 10:
            print(f"Sample indices: {indices}")
        else:
            print(f"Sample indices: {indices[:5]}...{indices[-5:]}")
        
        # 执行评估
        results = evaluator.evaluate_batch(
            nora_model=nora,
            unnormalizer=unnormalize,
            indices=indices,
            map_gripper=args.map_gripper_to_neg1_pos1,
            invert_gripper=args.invert_gripper,
            instruction_input=args.instruction
        )
        
        # 生成报告
        print("\n=== Generating Report ===")
        report = evaluator.generate_report(results)
        print(json.dumps(report, indent=2))
        
        # 保存结果
        print("\n=== Saving Results ===")
        evaluator.save_results(results)
        
        # 生成图表
        print("\n=== Generating Plots ===")
        try:
            evaluator.plot_results(results)
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
        
        print(f"\nEvaluation completed! Results saved to {args.output_dir}")
        
        # 打印简要统计
        if results['results']:
            times = results['times']
            errors = [r['l1_error'] for r in results['results']]
            print(f"\nQuick Summary:")
            print(f"- Samples processed: {len(results['results'])}")
            print(f"- Average inference time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
            print(f"- Average L1 error: {np.mean(errors):.4f} ± {np.std(errors):.4f}")


if __name__ == "__main__":
    main()