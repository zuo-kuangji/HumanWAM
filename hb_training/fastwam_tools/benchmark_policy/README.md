# B6 FastWAM 数据增强对照

在已分配的训练环境中生成两套逐节点命令。计划使用同一份已转换的 `b6v6` 数据、
ActionDiT 初始化、模型和 T5 缓存、每卡 batch、训练 seed 与总步数。唯一的训练变量是增强配方。
`unified` 组显式使用统一配方 v2，保留 5% 原图，进入增强后有 15% 概率选中一个强档。

```bash
python hb_training/fastwam_tools/hb_fastwam_compare.py \
  --python "$FASTWAM_PYTHON" --fastwam-source "$FASTWAM_SRC" \
  --data-root "$B6V6_DATA" --action-init "$ACTION_DIT_INIT" \
  --model-cache "$WAN_CACHE" --text-cache "$T5_CACHE" \
  --chosen "$FASTWAM_BATCH_RESULT/chosen.json" --output-root "$PAIR_RUNS" \
  --master-addr "$MASTER_ADDR" --machines 4 --steps 4000 --seed 42 > "$PAIR_PLAN"
```

`chosen.json` 与 `fastwam.ok` 来自同一轮 FastWAM batch 基准测试。默认四台机器，每台八卡。
计划输出 JSON，`runs.<配方>.ranks` 逐节点列出工作目录、环境变量和参数。
在各节点按对应 rank 先清除 `unset_env` 和 `unset_env_prefixes` 所列的集群启动变量，
再设置 `env` 并在 `cwd` 运行 `argv`。默认比较两个 `step_004000.pt`，
文件位于 `$PAIR_RUNS/{hb_v2,unified}/checkpoints/weights/`。保留每个 run 的 `config.yaml`、
`dataset_stats.json` 和对应 T5 embedding。

在 Humanoid_benchmark 的评估工作树中建立 `evaluation/policy/FastWAMB6Aug/`，复制本目录的
`model.py`，并写入以下 `deploy.yml`。所有路径填评估机器可访问的绝对路径，
run 的配置所引用的初始化权重及 T5 缓存也须可读。两组策略都使用训练 run 的原生 FastWAM 推理与归一化。

```yaml
policy_name: FastWAMB6Aug
python: /absolute/path/to/fastwam/bin/python
action_space: sonic_dfx6
checkpoints:
  hb_v2: /absolute/pair_runs/hb_v2/checkpoints/weights/step_004000.pt
  unified: /absolute/pair_runs/unified/checkpoints/weights/step_004000.pt
default_checkpoint: hb_v2
config:
  humanwam_root: /absolute/path/to/HumanWAM
  fastwam_source: /absolute/path/to/FastWAM
  model_cache: /absolute/path/to/wan_cache
```

用 B6 的 `v5_ready_start` 协议和锁定的 v5 `b6.yaml` 评估，结果中记录训练集为 B6-v6 和评估协议为 v5。
任务文字从 B6-v6 的 `meta/tasks.jsonl` 读取，使两组均命中训练使用的 T5 cache。
沿用 Humanoid_benchmark 配好的仿真环境。先对同一 seed 运行 `--dry-run`，核对布局和
`v5_recipe: true`。[B6 泛化协议](https://github.com/zuo-kuangji/Humanoid_benchmark/blob/5c8a0895457e49032b0e7c58528910c2549c5c5b/debug/tasks/b6/generalization_2026-11-17.md)

```bash
export PXR_WORK_THREAD_LIMIT=1
export PYTHONPATH=src:whole_body_controller:whole_body_controller/external_dependencies/unitree_sdk2_python:debug/newton:evaluation
for profile in hb_v2 unified; do
  for seed in $(seq 190000 190059); do
    .venv_sim/bin/python debug/newton/b6_policy_eval.py \
      --protocol v5_ready_start --task-config "$B6_V5_YAML" \
      --task-text "$B6V6_INSTRUCTION" --policy "FastWAMB6Aug/$profile" \
      --seed "$seed" --output "$B6_EVAL_OUT/$profile/normal/$seed" || exit 1
    .venv_sim/bin/python debug/newton/b6_gen_eval.py \
      --protocol v5_ready_start --task-config "$B6_V5_YAML" \
      --task-text "$B6V6_INSTRUCTION" --policy "FastWAMB6Aug/$profile" \
      --seed "$seed" --output "$B6_EVAL_OUT/$profile/generalization/$seed" || exit 1
  done
done
```

以 `policy_result.json` 里的成功指标为准，报告普通场景和 A–E 各类成功率以及成对变化。
两组使用相同 seed 顺序和每次查询的采样 seed。单独记录基础场景是否退化，
保留失败案例与运行报错，再据结果决定是否替换默认配方。
