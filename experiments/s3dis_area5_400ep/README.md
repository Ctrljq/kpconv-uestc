# S3DIS Area-5 400 Epoch Ablation Runs

This folder stores the 400 epoch S3DIS Area-5 ablation experiments.

## Official Main Experiments

Area split:

- Train: `Area_1`, `Area_2`, `Area_3`, `Area_4`, `Area_6`
- Validation/Test: `Area_5`
- `Area_7` is disabled in the official main experiments.

Run commands:

```bash
bash scripts/run_s3dis_experiment.sh E01_baseline_ce
bash scripts/run_s3dis_experiment.sh E02_attentiongate_ce
bash scripts/run_s3dis_experiment.sh E03_baseline_weightedce
bash scripts/run_s3dis_experiment.sh E04_attentiongate_weightedce
```

## Area_7 Extension Experiments

Area_7 is optional extra training data and must not be reported as an official Area-5 comparison.

```bash
bash scripts/run_s3dis_experiment.sh E05_ag_weightedce_area7_r10
bash scripts/run_s3dis_experiment.sh E06_ag_weightedce_area7_r5
```

`--area7-ratio 10` means official training areas as a group : Area_7 is `10:1`, so Area_7 is expected to provide about 9.1% of training samples.
Edit the files in `scripts/s3dis_experiments/` if you want to change epochs, GPU id, AttentionGate, loss type, or the Area_7 ratio.

## Per-Run Files

Each run directory should contain:

```text
run_info.md
run_info.json
parameters.txt
training.txt
val_IoUs.txt
console.log
launch_command.txt
metrics/
  best_metrics.json
  final_metrics.json
checkpoints/
  current_chkp.tar
  best_miou_chkp.tar
val_preds_400/
```

## Summary CSV

After training, summarize results into `summary.csv` with:

```text
experiment,use_attention_gate,loss_type,include_area7,area7_sampling_ratio,best_epoch,best_mIoU,final_mIoU,ceiling,floor,wall,beam,column,window,door,chair,table,bookcase,sofa,board,clutter
```
