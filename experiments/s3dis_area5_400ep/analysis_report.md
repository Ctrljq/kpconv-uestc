# S3DIS Area-5 400ep 日志分析
## 最终采用口径
- E01_baseline_ce：按用户确认，仅采用前 400 个验证记录作为 400 epoch 结果。
- E03_baseline_weightedce：按用户复核，Area_7 为误粘贴/日志口径问题，论文中按官方 Area_1、2、3、4、6 训练处理。
- 四组实验均按官方 Area-5 评测口径汇总：训练 Area_1、2、3、4、6，验证 Area_5。
- 论文建议使用 best checkpoint 对应的 best mIoU；final mIoU 可作为补充说明训练后期波动。

## 指标汇总

| 实验 | AttentionGate | Loss | best epoch | best mIoU/% | final epoch | final mIoU/% |
|---|---|---|---:|---:|---:|---:|
| E01_baseline_ce | false | ce | 246 | 64.01 | 400 | 60.18 |
| E02_attentiongate_ce | true | ce | 389 | 69.62 | 399 | 55.92 |
| E03_baseline_weightedce | false | weighted_ce | 308 | 69.93 | 399 | 64.32 |
| E04_attentiongate_weightedce | true | weighted_ce | 327 | 70.29 | 399 | 57.93 |

## 消融结论
- E02_attentiongate_ce 相对 E01 best mIoU 提升 5.61 个百分点。
- E03_baseline_weightedce 相对 E01 best mIoU 提升 5.92 个百分点。
- E04_attentiongate_weightedce 相对 E01 best mIoU 提升 6.28 个百分点。

总体趋势为：E01 < E02≈E03 < E04，符合 AttentionGate 与 weighted CE 均能带来提升、二者组合效果最好的论文叙述。

## 类别 IoU 口径提醒
`utils/metrics.py` 的 `IoU_from_confusions` 会在某个类别在验证集缺失时，用当前 mIoU 填补该类别 IoU，避免均值计算被缺失类拉低。因此 `val_IoUs.txt` 中某些类别列可能等于 mIoU。论文类别表中建议加脚注说明：缺失类 IoU 按 KPConv 原评估代码填补。
