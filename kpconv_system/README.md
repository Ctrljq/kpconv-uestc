# 基于 KPConv-AttentionGate 的室内点云语义分割系统

这个目录是独立的 Web 演示系统，不修改 `source_code/kpconv-uestc` 源码。系统支持上传 `.ply` 点云、选择 KPConv checkpoint、调用模型推理、生成彩色 `.ply`，并在网页中显示分割结果。

## 运行方式

```bash
cd /Users/zhangkaiqi/Documents/毕业设计/kpconv_system
python3 app.py
```

然后打开：

```text
http://127.0.0.1:8000
```

如果使用 uvicorn：

```bash
uvicorn app:app --reload --port 8000
```

## 依赖

系统依赖原 KPConv 环境中的 PyTorch、scikit-learn、NumPy 和 C++ wrappers，同时需要 Web 服务依赖：

```bash
pip install fastapi uvicorn python-multipart
```

如果导入 `cpp_wrappers` 失败，请先在 KPConv 源码目录编译：

```bash
cd /Users/zhangkaiqi/Documents/毕业设计/source_code/kpconv-uestc
bash cpp_wrappers/compile_wrappers.sh
```

## 权重放置方式

推荐保留 KPConv 训练日志结构：

```text
Log_xxx/
  parameters.txt
  checkpoints/
    current_chkp.tar
```

系统会扫描：

```text
/Users/zhangkaiqi/Documents/毕业设计/kpconv_system/weights
/Users/zhangkaiqi/Documents/毕业设计/source_code/kpconv-uestc/results
/root/autodl-tmp/s3dis_area5_400ep
```

如果只复制 checkpoint 而没有对应的 `parameters.txt`，系统会提示缺少配置文件。可以把完整的 `Log_xxx` 目录放到 `kpconv_system/weights` 下。

如果系统目录被移动到其他位置，或者 KPConv 源码不在默认位置，可以用环境变量指定：

```bash
export KPCONV_ROOT=/root/KPConv-PyTorch
export KPCONV_WEIGHT_DIRS=/root/autodl-tmp/s3dis_area5_400ep
export KPCONV_CLOUD_DIRS=/root/autodl-tmp/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/original_ply
python app.py
```

远程服务器访问时建议监听所有网卡：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 输入 PLY 要求

第一版仅支持二进制 `.ply`，字段至少包含：

```text
x, y, z
```

如果包含：

```text
red, green, blue
```

系统会使用颜色作为输入特征；如果没有颜色，系统会自动补零颜色特征继续推理。

## 输出结果

每次任务会在下面生成一个独立目录：

```text
outputs/{job_id}/
  colored.ply
  result.json
```

`colored.ply` 包含：

```text
x, y, z, red, green, blue, preds
```

其中 `preds` 是 S3DIS 13 类语义标签。

## 常见问题

- 页面提示未找到权重：把 `.tar/.pth/.pt` 权重和 `parameters.txt` 放到 `weights/Log_xxx/` 或 KPConv `results/Log_xxx/`。
- 页面提示 PLY 不是二进制：当前复用 KPConv 的 `read_ply`，不支持 ASCII PLY。
- 点云很大时速度慢：系统会自动压缩推理点数，再用最近邻把预测标签回填到原始点。
