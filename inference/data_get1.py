from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch

# 1) 加载数据（你已换成 pyav，保留）
ds = LeRobotDataset(
    repo_id="xuanyuanj/so101_v3",
    root="/home/luka/Nora_lerobot/so101_pick_the_cube",
    video_backend="pyav",
)

# 2) 通用结构打印器：同时识别 numpy 和 torch 张量
def describe(obj, prefix=""):
    def shape_dtype(x):
        if isinstance(x, np.ndarray):
            return f"ndarray{list(x.shape)} {x.dtype}"
        if torch.is_tensor(x):
            return f"tensor{list(x.shape)} {x.dtype}"
        return type(x).__name__
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k} -> {shape_dtype(v)}")
            describe(v, prefix + "  ")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj[:5]):  # 只展开前5个
            print(f"{prefix}[{i}] -> {shape_dtype(v)}")
            describe(v, prefix + "  ")

sample = ds[0]
print("=== Top-level keys:", list(sample.keys()))
print("=== Full structure (first sample) ===")
describe(sample)

# 3) 自动查找一个“看起来像图像”的键 和 一个“看起来像状态/动作”的键
def find_image_and_state_keys(d):
    img_key, st_key = None, None
    stack = [([], d)]
    def is_image(x):
        # HxWxC 或 CxHxW；numpy 或 torch 都可
        if isinstance(x, np.ndarray) and x.ndim == 3 and (x.shape[2] in (1,3,4) or x.shape[0] in (1,3,4)):
            return True
        if torch.is_tensor(x) and x.ndim == 3 and (x.shape[2] in (1,3,4) or x.shape[0] in (1,3,4)):
            return True
        return False
    def is_state(x):
        # 1D/2D 数值（动作/关节/位姿等）
        if isinstance(x, np.ndarray) and x.ndim in (1,2) and x.dtype.kind in "fi":
            return True
        if torch.is_tensor(x) and x.ndim in (1,2) and str(x.dtype).startswith("torch."):
            return True
        return False

    while stack:
        path, node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                stack.append((path+[k], v))
        elif isinstance(node, (list, tuple)):
            # 若是序列，尝试第0项
            if node:
                stack.append((path+[0], node[0]))
        else:
            if img_key is None and is_image(node):
                img_key = path
            if st_key is None and is_state(node):
                st_key = path
        if img_key and st_key:
            break
    return img_key, st_key

img_path, state_path = find_image_and_state_keys(sample)
print("\n=== Detected paths ===")
print("Image path:", img_path)
print("State/Action path:", state_path)

# 4) 按路径安全取值
def get_by_path(d, path):
    cur = d
    for p in path:
        if isinstance(p, int):
            cur = cur[p]
        else:
            cur = cur[p]
    return cur

if img_path is None or state_path is None:
    raise RuntimeError("未自动检测到图像或状态键。请把上面的 Full structure 输出发我，我来帮你精确定位。")

img = get_by_path(sample, img_path)
state = get_by_path(sample, state_path)

# 5) 统一成常用格式
import torchvision
from torchvision.transforms.functional import to_pil_image

if isinstance(img, np.ndarray):
    # numpy: HxWxC 或 CxHxW 都能转
    img_pil = to_pil_image(img)
    img_chw = torchvision.transforms.functional.to_tensor(img)  # [C,H,W], float32
elif torch.is_tensor(img):
    # torch: 先转 numpy 再转 PIL，或直接规范到 [C,H,W]
    if img.ndim == 3 and img.shape[0] in (1,3,4):
        img_chw = img.float()
        img_pil = to_pil_image(img_chw.cpu())
    elif img.ndim == 3 and img.shape[-1] in (1,3,4):
        img_np = img.cpu().numpy()
        img_pil = to_pil_image(img_np)
        img_chw = torchvision.transforms.functional.to_tensor(img_np)
    else:
        raise ValueError(f"Unrecognized image shape: {tuple(img.shape)}")
else:
    raise TypeError(f"Unsupported image type: {type(img)}")

state_t = torch.as_tensor(state).float()
if state_t.ndim == 2:
    # 如果是序列，默认取最后一帧（按需改成你想要的 t）
    state_t = state_t[-1]

print("\n=== Final tensors for inference ===")
print("Image PIL size:", img_pil.size)
print("Image tensor:", img_chw.shape, img_chw.dtype)
print("State tensor:", state_t.shape, state_t.dtype)
