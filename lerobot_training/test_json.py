# from pathlib import Path
# import json

# INFO_PATH = "/data4/hydeng/nora/lerobot_so101_v3/meta/info.json"  # 假设固定文件名

# def load_json(path: Path) -> dict:
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def load_info(local_dir: Path) -> dict:
#     info = load_json(INFO_PATH)
#     for ft in info["features"].values():
#         ft["shape"] = tuple(ft["shape"])
#     return info

# info = load_info(local_dir="home")

from pathlib import Path
import json
from typing import Any, Dict, Union

# 你的绝对路径（如果存在，就优先用）
INFO_PATH = Path("/data4/hydeng/nora/lerobot_so101_v3/meta/info.json")

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_tuple_if_list(x):
    return tuple(x) if isinstance(x, list) else x

def _resolve_info_path(local_dir: Union[str, Path]) -> Path:
    """
    - 如果 INFO_PATH 是个存在的绝对路径，就直接用它；
    - 否则尝试使用 local_dir / 'meta/info.json'。
    """
    if INFO_PATH.is_absolute() and INFO_PATH.exists():
        return INFO_PATH
    local_dir = Path(local_dir)
    candidate = local_dir / "meta" / "info.json"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"info.json not found at {INFO_PATH} or {candidate}")

def load_info(local_dir: Union[str, Path]) -> Dict[str, Any]:
    info_path = _resolve_info_path(local_dir)
    info = load_json(info_path)

    if "features" not in info:
        raise KeyError(f"'features' not found in {info_path}")

    feats = info["features"]

    # 情况A：features 是 dict
    if isinstance(feats, dict):
        for k, v in feats.items():
            if isinstance(v, dict) and "shape" in v:
                v["shape"] = _to_tuple_if_list(v["shape"])
        return info

    # 情况B：features 是 list
    if isinstance(feats, list):
        for i, v in enumerate(feats):
            if isinstance(v, dict) and "shape" in v:
                v["shape"] = _to_tuple_if_list(v["shape"])
        return info

    # 其他类型：给出清晰错误
    raise TypeError(
        f"'features' must be a dict or list, but got {type(feats).__name__} from {info_path}"
    )

# —— 可选：快速诊断谁是字符串 / 非 dict —— #
def debug_features_structure(local_dir: Union[str, Path]) -> None:
    info_path = _resolve_info_path(local_dir)
    info = load_json(info_path)
    feats = info.get("features", None)
    if feats is None:
        print("No 'features' key in info.json")
        return

    if isinstance(feats, dict):
        print("features is a dict:")
        for k, v in feats.items():
            print(f"  - {k}: {type(v).__name__} | keys={list(v.keys()) if isinstance(v, dict) else 'N/A'}")
    elif isinstance(feats, list):
        print("features is a list:")
        for i, v in enumerate(feats):
            print(f"  - idx {i}: {type(v).__name__} | keys={list(v.keys()) if isinstance(v, dict) else 'N/A'}")
    else:
        print(f"features is {type(feats).__name__}, value preview: {str(feats)[:200]}")


info = load_info(local_dir="home")
print("Loaded info:", info)