from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("KPCONV_PROJECT_ROOT", BASE_DIR.parents[0])).expanduser().resolve()
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
WEIGHTS_DIR = BASE_DIR / "weights"
KPCONV_RESULTS_DIR = Path(
    os.environ.get("KPCONV_RESULTS_DIR", PROJECT_ROOT / "source_code" / "kpconv-uestc" / "results")
).expanduser().resolve()
DEFAULT_REMOTE_EXPERIMENTS_DIR = Path("/root/autodl-tmp/s3dis_area5_400ep")

for folder in (UPLOAD_DIR, OUTPUT_DIR, WEIGHTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="KPConv-AttentionGate Point Cloud Segmentation System")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/weights")
def api_weights() -> dict:
    roots = _weight_roots()
    return {"weights": _discover_weights(*roots)}


@app.post("/api/segment")
async def api_segment(file: UploadFile = File(...), weight_path: str = Form(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".ply"):
        raise HTTPException(status_code=400, detail="请上传 .ply 点云文件。")

    checkpoint = _validate_weight_path(weight_path)
    job_id = uuid.uuid4().hex[:12]
    job_upload_dir = UPLOAD_DIR / job_id
    job_output_dir = OUTPUT_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_upload_dir / _safe_filename(file.filename)
    with input_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        segment_ply = _load_inference_functions()
        result = segment_ply(input_path, checkpoint, job_output_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "job_id": job_id,
        "colored_ply_url": f"/outputs/{job_id}/colored.ply",
        "result_url": f"/outputs/{job_id}/result.json",
        "result": result,
    }


@app.get("/outputs/{job_id}/colored.ply")
def colored_ply(job_id: str) -> FileResponse:
    path = OUTPUT_DIR / job_id / "colored.ply"
    if not path.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在。")
    return FileResponse(path, filename="colored.ply")


@app.get("/outputs/{job_id}/result.json")
def result_json(job_id: str) -> FileResponse:
    path = OUTPUT_DIR / job_id / "result.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="结果文件不存在。")
    return FileResponse(path)


def _safe_filename(name: str) -> str:
    keep = [c if c.isalnum() or c in "._-" else "_" for c in name]
    safe = "".join(keep).strip("._")
    return safe or "input.ply"


def _validate_weight_path(raw_path: str) -> Path:
    try:
        path = Path(raw_path).expanduser().resolve()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="权重路径无效。") from exc

    allowed_roots = [root.resolve() for root in _weight_roots()]
    if not any(_is_relative_to(path, root) for root in allowed_roots):
        raise HTTPException(status_code=400, detail="权重必须位于系统允许扫描的权重目录。")
    if not path.exists() or path.suffix.lower() not in {".tar", ".pth", ".pt"}:
        raise HTTPException(status_code=400, detail="请选择存在的 .tar/.pth/.pt 权重文件。")
    return path


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _discover_weights(*roots: Path) -> list[dict]:
    weight_files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.tar", "*.pth", "*.pt"):
            weight_files.extend(root.rglob(pattern))

    items = []
    for path in sorted(set(weight_files)):
        try:
            config_path = _find_config_for_checkpoint(path)
            status = "ready"
            message = "found parameters.txt"
        except FileNotFoundError as exc:
            config_path = None
            status = "missing_config"
            message = str(exc)

        items.append(
            {
                "id": str(path.resolve()),
                "name": _display_weight_name(path),
                "path": str(path.resolve()),
                "status": status,
                "message": message,
                "config_path": str(config_path) if config_path else None,
            }
        )
    return items


def _weight_roots() -> list[Path]:
    roots = [WEIGHTS_DIR.resolve(), KPCONV_RESULTS_DIR.resolve()]
    if DEFAULT_REMOTE_EXPERIMENTS_DIR.exists():
        roots.append(DEFAULT_REMOTE_EXPERIMENTS_DIR.resolve())

    extra = os.environ.get("KPCONV_WEIGHT_DIRS", "")
    for raw in extra.split(":"):
        raw = raw.strip()
        if raw:
            roots.append(Path(raw).expanduser().resolve())

    deduped = []
    seen = set()
    for root in roots:
        if str(root) not in seen:
            deduped.append(root)
            seen.add(str(root))
    return deduped


def _display_weight_name(path: Path) -> str:
    parts = path.resolve().parts
    if "checkpoints" in parts:
        idx = parts.index("checkpoints")
        if idx > 0:
            return f"{parts[idx - 1]} / checkpoints / {path.name}"
    return path.name


def _find_config_for_checkpoint(checkpoint_path: Path) -> Path:
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


def _load_inference_functions():
    try:
        from inference import segment_ply
    except ModuleNotFoundError as exc:
        missing = exc.name or "依赖"
        raise HTTPException(
            status_code=500,
            detail=(
                f"后端推理依赖缺失：{missing}。如果只是打开网页，请确认 FastAPI 服务已启动；"
                "如果要真实分割，请在安装了 PyTorch、scikit-learn 和 KPConv C++ wrappers 的环境中运行。"
            ),
        ) from exc
    return segment_ply


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
