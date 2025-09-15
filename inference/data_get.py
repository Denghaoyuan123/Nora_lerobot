import torch
import torchvision
from torchvision.transforms.functional import to_pil_image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    repo_id="xuanyuanj/so101_v3",
    root="/home/luka/Nora_lerobot/so101_pick_the_cube",
    video_backend="pyav",
)

ex = ds[0]

img_chw = ex["observation.images.front"]          # [C,H,W], float32 (已是0~1或数据集定义的范围)
state   = ex["observation.state"].float()         # [D]
lang    = ex["task"]                               # str
# 可选：标签
# target_action = ex["action"].float()            # [D]

# 如果你的 VLM 需要 PIL（比如 apply_chat_template+processor）
img_pil = to_pil_image(img_chw)  # 会自动将 CHW float 转为 PIL(RGB)

print("PIL:", img_pil.size, "CHW:", img_chw.shape, "state:", state.shape, "lang:", lang)
