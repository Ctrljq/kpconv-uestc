from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.neighbors import KDTree


BASE_DIR = Path(__file__).resolve().parent
KPCONV_ROOT = Path(os.environ.get("KPCONV_ROOT", "")).expanduser()
if not str(KPCONV_ROOT):
    if (BASE_DIR.parent / "models").exists() and (BASE_DIR.parent / "datasets").exists():
        KPCONV_ROOT = BASE_DIR.parent
    else:
        KPCONV_ROOT = BASE_DIR.parents[1] / "source_code" / "kpconv-uestc"
KPCONV_ROOT = KPCONV_ROOT.resolve()
PROJECT_ROOT = KPCONV_ROOT.parent

if str(KPCONV_ROOT) not in sys.path:
    sys.path.insert(0, str(KPCONV_ROOT))

from datasets.common import PointCloudDataset, grid_subsampling  # noqa: E402
from datasets.S3DIS import S3DISCustomBatch  # noqa: E402
from models.architectures import KPFCNN  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.ply import read_ply, write_ply  # noqa: E402


S3DIS_LABEL_TO_NAMES = {
    0: "ceiling",
    1: "floor",
    2: "wall",
    3: "beam",
    4: "column",
    5: "window",
    6: "door",
    7: "chair",
    8: "table",
    9: "bookcase",
    10: "sofa",
    11: "board",
    12: "clutter",
}

S3DIS_COLORS = np.array(
    [
        [145, 184, 255],
        [112, 173, 71],
        [255, 192, 0],
        [180, 95, 6],
        [160, 160, 160],
        [91, 155, 213],
        [237, 125, 49],
        [112, 48, 160],
        [255, 102, 178],
        [68, 114, 196],
        [165, 105, 189],
        [0, 176, 240],
        [127, 127, 127],
    ],
    dtype=np.uint8,
)


@dataclass
class LoadedModel:
    checkpoint_path: Path
    config_path: Path
    config: Config
    net: KPFCNN
    device: torch.device


class SinglePlyDatasetAdapter(PointCloudDataset):
    """Small adapter that exposes KPConv's segmentation input builder."""

    def __init__(self, config: Config):
        super().__init__("S3DIS-single-ply")
        self.config = config
        self.label_to_names = S3DIS_LABEL_TO_NAMES
        self.init_labels()
        self.ignored_labels = np.array([], dtype=np.int32)
        self.neighborhood_limits = []


_MODEL_CACHE: Dict[str, LoadedModel] = {}


def discover_weights(*roots: Path) -> List[dict]:
    weight_files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.tar", "*.pth", "*.pt"):
            weight_files.extend(root.rglob(pattern))

    items = []
    for path in sorted(set(weight_files)):
        try:
            config_path = find_config_for_checkpoint(path)
            status = "ready"
            message = "found parameters.txt"
        except FileNotFoundError as exc:
            config_path = None
            status = "missing_config"
            message = str(exc)

        items.append(
            {
                "id": str(path.resolve()),
                "name": display_weight_name(path),
                "path": str(path.resolve()),
                "status": status,
                "message": message,
                "config_path": str(config_path) if config_path else None,
            }
        )
    return items


def display_weight_name(path: Path) -> str:
    parts = path.resolve().parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx > 0:
            return f"{parts[idx - 1]} / checkpoints / {path.name}"
    return path.name


def find_config_for_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    candidates = [
        checkpoint_path.parent / "parameters.txt",
        checkpoint_path.parent.parent / "parameters.txt",
        checkpoint_path.parent.parent.parent / "parameters.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "没有找到 checkpoint 对应的 parameters.txt。请把权重放在训练日志目录下，"
        "例如 Log_xxx/checkpoints/current_chkp.tar，并保留 Log_xxx/parameters.txt。"
    )


def load_model(checkpoint_path: Path) -> LoadedModel:
    checkpoint_path = checkpoint_path.resolve()
    cache_key = str(checkpoint_path)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    config_path = find_config_for_checkpoint(checkpoint_path)
    config = Config()
    config.load(str(config_path.parent))
    config.dataset = "S3DIS"
    config.dataset_task = "cloud_segmentation"
    config.num_classes = len(S3DIS_LABEL_TO_NAMES)
    config.input_threads = 0
    config.augment_noise = 0.0
    config.augment_color = 1.0

    label_values = np.array(sorted(S3DIS_LABEL_TO_NAMES.keys()), dtype=np.int32)
    ignored_labels = np.array([], dtype=np.int32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)

    net = KPFCNN(config, label_values, ignored_labels)
    try:
        net.load_state_dict(state)
    except RuntimeError as exc:
        if getattr(config, "use_attention_gate", True) and _looks_like_baseline_state_dict(state):
            config.use_attention_gate = False
            net = KPFCNN(config, label_values, ignored_labels)
            net.load_state_dict(state)
        else:
            raise RuntimeError(
                "模型权重与 parameters.txt 中的网络结构不匹配。请确认选择的是同一次训练生成的 "
                "checkpoint 和 parameters.txt。原始错误: " + str(exc)
            ) from exc
    net.to(device)
    net.eval()

    loaded = LoadedModel(checkpoint_path, config_path, config, net, device)
    _MODEL_CACHE[cache_key] = loaded
    return loaded


def segment_ply(
    input_ply: Path,
    checkpoint_path: Path,
    output_dir: Path,
    max_points: int = 65000,
) -> dict:
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = read_ply(str(input_ply))
    points = _read_points(data)
    colors, had_color = _read_colors(data, points.shape[0])

    loaded = load_model(checkpoint_path)
    run_points, run_colors, run_indices, messages = _prepare_runtime_points(
        points, colors, loaded.config, max_points
    )

    logits = _predict_points(loaded, run_points, run_colors)
    run_preds = np.argmax(logits, axis=1).astype(np.int32)

    if run_points.shape[0] == points.shape[0] and np.array_equal(run_indices, np.arange(points.shape[0])):
        preds = run_preds
    else:
        tree = KDTree(run_points, leaf_size=20)
        nearest = tree.query(points, k=1, return_distance=False).reshape(-1)
        preds = run_preds[nearest]
        messages.append(
            f"输入点数较多，系统对 {run_points.shape[0]} 个代表点推理，并用最近邻回填到 {points.shape[0]} 个原始点。"
        )

    colorized = S3DIS_COLORS[np.clip(preds, 0, len(S3DIS_COLORS) - 1)]
    colored_path = output_dir / "colored.ply"
    write_ply(
        str(colored_path),
        [points.astype(np.float32), colorized.astype(np.uint8), preds.astype(np.int32)],
        ["x", "y", "z", "red", "green", "blue", "preds"],
    )

    stats = _class_statistics(preds)
    result = {
        "input_file": str(input_ply),
        "checkpoint": str(checkpoint_path),
        "config": str(loaded.config_path),
        "colored_ply": str(colored_path),
        "num_points": int(points.shape[0]),
        "num_inference_points": int(run_points.shape[0]),
        "had_color": bool(had_color),
        "messages": messages,
        "classes": stats,
        "elapsed_seconds": round(time.time() - start, 3),
    }
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _read_points(data: np.ndarray) -> np.ndarray:
    names = data.dtype.names or ()
    required = ["x", "y", "z"]
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"PLY 缺少坐标字段: {', '.join(missing)}")
    return np.vstack((data["x"], data["y"], data["z"])).T.astype(np.float32)


def _read_colors(data: np.ndarray, n: int) -> Tuple[np.ndarray, bool]:
    names = data.dtype.names or ()
    if all(name in names for name in ("red", "green", "blue")):
        colors = np.vstack((data["red"], data["green"], data["blue"])).T.astype(np.float32)
        if colors.max(initial=0) > 1.0:
            colors /= 255.0
        return np.clip(colors, 0.0, 1.0), True
    return np.zeros((n, 3), dtype=np.float32), False


def _prepare_runtime_points(
    points: np.ndarray,
    colors: np.ndarray,
    config: Config,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    messages: List[str] = []
    dl = float(getattr(config, "first_subsampling_dl", 0.03) or 0.03)

    if points.shape[0] > max_points:
        sub_points, sub_colors = grid_subsampling(points, features=colors, sampleDl=dl)
        sub_points = sub_points.astype(np.float32)
        sub_colors = sub_colors.astype(np.float32)
        if sub_points.shape[0] > max_points:
            rng = np.random.default_rng(42)
            chosen = np.sort(rng.choice(sub_points.shape[0], size=max_points, replace=False))
            sub_points = sub_points[chosen]
            sub_colors = sub_colors[chosen]
        tree = KDTree(points, leaf_size=20)
        run_indices = tree.query(sub_points, k=1, return_distance=False).reshape(-1)
        messages.append(f"已按 {dl:.3f}m 网格和 {max_points} 点上限压缩推理点数。")
        return sub_points, sub_colors, run_indices.astype(np.int64), messages

    return points.astype(np.float32), colors.astype(np.float32), np.arange(points.shape[0]), messages


def _predict_points(loaded: LoadedModel, points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    adapter = SinglePlyDatasetAdapter(loaded.config)

    center = np.mean(points, axis=0, keepdims=True).astype(np.float32)
    centered = (points - center).astype(np.float32)
    heights = points[:, 2:3].astype(np.float32)
    features_4 = np.hstack((colors.astype(np.float32), heights)).astype(np.float32)

    stacked_features = np.ones_like(centered[:, :1], dtype=np.float32)
    if loaded.config.in_features_dim == 1:
        pass
    elif loaded.config.in_features_dim == 4:
        stacked_features = np.hstack((stacked_features, features_4[:, :3])).astype(np.float32)
    elif loaded.config.in_features_dim == 5:
        stacked_features = np.hstack((stacked_features, features_4)).astype(np.float32)
    else:
        raise ValueError(f"当前系统仅支持 in_features_dim 为 1、4 或 5，当前为 {loaded.config.in_features_dim}")

    labels = np.zeros((points.shape[0],), dtype=np.int64)
    lengths = np.array([points.shape[0]], dtype=np.int32)
    input_list = adapter.segmentation_inputs(centered, stacked_features, labels, lengths)
    input_list += [
        np.ones((1,), dtype=np.float32),
        np.eye(3, dtype=np.float32).reshape(1, 3, 3),
        np.array([0], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.arange(points.shape[0], dtype=np.int32),
    ]
    batch = S3DISCustomBatch([input_list])
    batch.to(loaded.device)

    with torch.no_grad():
        outputs = loaded.net(batch, loaded.config)
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
    return probs


def _class_statistics(preds: np.ndarray) -> List[dict]:
    total = max(int(preds.shape[0]), 1)
    stats = []
    for label, name in S3DIS_LABEL_TO_NAMES.items():
        count = int(np.sum(preds == label))
        stats.append(
            {
                "label": int(label),
                "name": name,
                "count": count,
                "percent": round(100.0 * count / total, 2),
                "color": S3DIS_COLORS[label].tolist(),
            }
        )
    return stats


def _looks_like_baseline_state_dict(state: dict) -> bool:
    keys = list(state.keys())
    has_attention = any("attention_gates" in key for key in keys)
    return not has_attention
