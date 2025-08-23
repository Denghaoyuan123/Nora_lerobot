# data_helper.py
def patch_lerobot_load_info():
    """Normalize info.json so that `features` is always a dict(name->spec)."""
    import os, json
    import lerobot.datasets.utils as lru
    import lerobot.datasets.lerobot_dataset as lrd

    if getattr(lru, "_patched_features_as_dict", False):
        # 确保 lerobot_dataset 也用到同一个函数
        if getattr(lrd, "load_info", None) is not lru.load_info:
            lrd.load_info = lru.load_info
        return

    def _load_info_patched(root: str):
        path = os.path.join(root, "info.json")
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)

        feats = info.get("features")

        # 如果被二次序列化成字符串，先反序列化
        if isinstance(feats, str):
            try:
                feats = json.loads(feats)
            except Exception:
                # 极端情况：整个 features 就是一个字符串
                feats = {"task": feats}

        # 统一转成 dict(name->spec)
        if isinstance(feats, list):
            feats_dict = {}
            for spec in feats:
                if isinstance(spec, str):
                    # 列表里出现纯字符串就跳过或自定义处理
                    continue
                spec = dict(spec)
                key = spec.pop("key", None)
                if key is None:
                    raise ValueError("features as list requires each item to have a 'key' field")
                # 规范 shape
                shp = spec.get("shape")
                if isinstance(shp, list):
                    spec["shape"] = tuple(shp)
                feats_dict[key] = spec
            feats = feats_dict
        elif isinstance(feats, dict):
            norm = {}
            for k, spec in feats.items():
                if isinstance(spec, str):
                    # 例如 "task": "touch box" -> 合法 string 特征
                    spec = {"dtype": "string", "shape": (1,), "names": None, "default": spec}
                else:
                    spec = dict(spec)
                    shp = spec.get("shape")
                    if isinstance(shp, list):
                        spec["shape"] = tuple(shp)
                norm[k] = spec
            feats = norm
        else:
            feats = {}

        info["features"] = feats
        return info

    lru.load_info = _load_info_patched
    lrd.load_info = _load_info_patched
    lru._patched_features_as_dict = True
    print("[lerobot] load_info() patched (features -> dict(name->spec)).")


def patch_lerobot_skip_hub():
    """禁用 LeRobot 的 Hub 版本查询，避免离线时访问 huggingface.co。"""
    import lerobot.datasets.utils as lru
    import lerobot.datasets.lerobot_dataset as lrd

    def _get_safe_version_offline(repo_id, revision=None):
        return revision or "local"

    lru.get_safe_version = _get_safe_version_offline
    if getattr(lrd, "get_safe_version", None) is not _get_safe_version_offline:
        lrd.get_safe_version = _get_safe_version_offline
    print("[lerobot] get_safe_version() patched -> offline/local")
