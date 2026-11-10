# Fast-WAM 原生维护入口

源码来自 [官方 FastWAM](https://github.com/yuantianyuan01/FastWAM)，固定 commit
`7faa71108368fbb3b6885649f112af607427a2d4`。保留 submodule 中的 MIT License
及第三方文件许可。此入口维护官方 Gaussian FM，不含我们历史实验中的 A2A、
endpoint loss、sparse-joint 等修改。以后加入这些方法需独立 patch、测试与 checkpoint。

## 环境与源码

使用独立环境。官方要求 Python>=3.10，README 示例为 Python3.10、
Torch2.7.1+cu128 / torchvision0.22.1+cu128，完整依赖以该 commit 的
`pyproject.toml` 为准，其中 Transformers4.49.0、Accelerate1.12.0、DeepSpeed0.18.7。
不要直接覆盖 GR00T、OpenPI 或 ImageWAM 的环境。

```bash
export HUMANWAM_ROOT=/absolute/path/to/HumanWAM
cd "$HUMANWAM_ROOT"
python scripts/humanwam/bootstrap.py --only fastwam
# 在为 FastWAM 准备的环境内，先按官方 README 安装对应 Torch/CUDA。
python -m pip install -e third_party/FastWAM
python -m pip install -e packages/humanwam
```

导入 adapter、`humanwam list` 与构造 policy 不导入 Torch，也不加载或下载权重。
首次 `predict` 才加载一次模型。推理前先验证图像、state、统计维度和可选SHA；
缺少本地基础资产时直接报错。适配禁止 native 的自动下载。

## 权重与基础资产

官方任务权重与统计来自 [yuanty/fastwam](https://huggingface.co/yuanty/fastwam)，
已核验 revision `8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`。

| Profile | 权重与对应统计文件 | 动作 / state / horizon | 当前图像 |
|---|---|---|---|
| `libero_uncond_2cam224` | `libero_uncond_2cam224.pt` + `libero_uncond_2cam224_dataset_stats.json` | 7 / 8 / 32 | base、wrist，各224²，横拼224×448 |
| `robotwin_uncond_3cam_384` | `robotwin_uncond_3cam_384.pt` + `robotwin_uncond_3cam_384_dataset_stats.json` | 14 / 14 / 32 | head置上方320×256，左右腕置下方各160×128，合384×320 |

两个 checkpoint 分别为12,041,735,140和12,041,813,092 bytes。LIBERO权重SHA256
`1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579`，统计SHA256
`30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`。
RoboTwin权重SHA256
`776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63`，统计SHA256
`7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095`。

只有配置了SHA才执行该文件的哈希校验，示例配置已填LIBERO官方两项SHA。
使用自己训练的同结构模型时，显式换成本次 checkpoint 和统计的SHA；维度相同不代表
统计、相机或控制语义相同。adapter另检查完整MoT及proprio权重的key和shape，
拒绝 native permissive loader 允许的缺失权重与 `dit`-only 视频 checkpoint。

任务 `.pt` 不含 VAE / T5。推理还需下面的原生资产，按HF repository目录放在
`FASTWAM_MODEL_CACHE` 下：

```text
FASTWAM_MODEL_CACHE/
  Wan-AI/Wan2.2-TI2V-5B/
    Wan2.2_VAE.pth
    models_t5_umt5-xxl-enc-bf16.pth
  Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
    tokenizer.json
    tokenizer_config.json
    ...其余原生 tokenizer 文件
```

官方工厂支持原始Wan文件和DiffSynth转换文件。本入口明确使用
`redirect_common_files=false` 的原始文件，因2026-11-09检查默认重定向仓库
`DiffSynth-Studio/Wan-Series-Converted-Safetensors` 返回HTTP401。
公开 [Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) revision
`921dbaf3f1674a56f47e83fb80a34bac8a8f203e` 与
[Wan2.1 tokenizer](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) revision
`37ec512624d61f7aa208f7ea8140a131f93afc9a` 已完成metadata核验。
VAE约2.82GB，T5约11.36GB；它们并未在本次集成中下载。

以下是供操作者执行的下载命令，本次未执行大文件下载：

```bash
hf download yuanty/fastwam \
  libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json \
  --revision 8eaceeb24c3cc92ff2a9c9a9d266a4941b836705 --local-dir /models/fastwam-release
hf download Wan-AI/Wan2.2-TI2V-5B \
  Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir "$FASTWAM_MODEL_CACHE/Wan-AI/Wan2.2-TI2V-5B"
hf download Wan-AI/Wan2.1-T2V-1.3B --include 'google/umt5-xxl/*' \
  --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --local-dir "$FASTWAM_MODEL_CACHE/Wan-AI/Wan2.1-T2V-1.3B"
```

部署完整任务权重时，原生工厂跳过基础video/action DiT初始化，因此无需额外下载
Wan DiT分片或生成ActionDiT initializer。新训练则需要Wan DiT及initializer。

## 离线预测与服务

```bash
export FASTWAM_CHECKPOINT=/models/fastwam-release/libero_uncond_2cam224.pt
export FASTWAM_STATS=/models/fastwam-release/libero_uncond_2cam224_dataset_stats.json
export FASTWAM_MODEL_CACHE=/models/wan-assets
humanwam predict --config configs/humanwam/fastwam-native.json \
  --observation /data/current_observation.npz --metadata /data/instruction.json \
  --output /data/prediction-new.json
humanwam serve --config configs/humanwam/fastwam-native.json --dry-run
```

NPZ包含 `images`，形状为 `[2,H,W,3]`，uint8 RGB；以及raw `state[8]`。
JSON metadata为 `{"prompt":"put the bowl on the plate"}`，只填任务指令，
adapter会使用官方固定prompt模板。LIBERO state顺序为EEF position3、
axis-angle3、gripper qpos2，不能直接传四元数形式或归一化state。
图像应已采用官方camera朝向，adapter不做额外翻转。

返回32×7未归一化动作，六维EEF增量及LIBERO gripper convention，默认按官方评测
二值化gripper。RoboTwin切换profile、匹配checkpoint/statistics及SHA，输入当前
`images=[head,left_wrist,right_wrist]` 和原始14D关节state，返回32×14关节动作。
它不接受过去动作，不把第二视角解释为未来图像，不生成G1/H2/SONIC动作。

调用的原生API是 `FastWAM.infer_action(prompt,input_image,action_horizon,proprio,...)`。
图像拼接、state/action normalizer与LIBERO gripper处理对应官方eval/deploy实现。
原版发布权重使用 `sigma_shift=5.0`；最新上游新训练默认action shift1.0，不能混用。
`compile_action_infer`在本例默认关闭，可在相应GPU验证后开启。

官方没有独立HTTP policy server。本仓库的serve是通用JSON/NPZ HTTP入口，
默认绑定127.0.0.1:8094；不是SONIC actuator接口，也不执行机器人动作。

官方当前state经 `proprio_encoder` 追加到text context，并进入视觉与动作计算。
所以这是原生state-conditioned Fast-WAM，不符合ImageWAM中“state只进入动作专家”
的独立变体约束。此维护入口保留这一原生行为，没有暗中修改监督或attention。

## 新训练与初始化

原生训练入口 `scripts/train.py` 使用Hydra，分布式配置为ZeRO1。LIBERO recipe
每卡batch16、累积1、10epochs，在8卡下global128；RoboTwin原生recipe为5epochs。
这些是上游默认值，不能直接解释成G1/H2任务的已验证设置。

从基础开始时下载Wan2.2的 `diffusion_pytorch_model*.safetensors`，三个分片合约20GB。
准备新的临时model YAML副本，将 `redirect_common_files` 设为false，再运行：

```bash
cd "$HUMANWAM_ROOT/third_party/FastWAM"
export DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_MODEL_CACHE"
export DIFFSYNTH_SKIP_DOWNLOAD=true
python scripts/preprocess_action_dit_backbone.py \
  --model-config /configs/fastwam-original-assets.yaml \
  --output "$FASTWAM_ACTION_INIT" --device cpu --dtype bfloat16
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4 \
  model.redirect_common_files=false \
  "data.train.dataset_dirs=[\"$FASTWAM_DATA_ROOT\"]" \
  "data.train.text_embedding_cache_dir=$FASTWAM_TEXT_CACHE"
cd "$HUMANWAM_ROOT"
humanwam train --config configs/humanwam/fastwam-train.json --dry-run
```

initializer是从Wan video DiT线性插值并缩放得到的动作共享backbone，跳过动作专用层，
不是训练过的LIBERO动作策略。原脚本遇到未解析action_dim会默认7，定制动作空间时
必须先显式解析自己的YAML，不依赖该默认值。新任务使用自己的数据、统计和text缓存。
默认训练配置未替你创建G1/H2数据映射或新动作头。

训练JSON还需设置 `FASTWAM_OUTPUT`、`FASTWAM_DATA_ROOT`、`FASTWAM_ACTION_INIT`、
`FASTWAM_TEXT_CACHE`。命令构建器保留上游cwd和配置层级，使用argv而非shell，
默认拒绝覆盖已有非空output目录。只构建命令不申请GPU或启动训练。

训练入口仅接受`method="gaussian_fm"`，其他方法明确报错。
顶层`profile`、`action_schema`、`action_dim`、`state_dim`和`horizon`是推理配置字段，
即使数值与原任务相符也不能放入训练options，以免被静默忽略。
训练配方由`task`和原生Hydra `overrides`确定。自定义overrides仍会转交原生入口，
但不因此获得新的受支持profile；变更动作/state维度或时间窗口，还需自己的数据映射、
统计、初始化及对应验证。当前没有G1/SONIC76/state44训练配方。

## 本次验证边界

已完成源码固定、公开HF metadata/两份小统计核验、CPU契约测试、真实Hydra配置compose
及原生 `infer_action` 参数签名静态核对。CPU测试使用注入的非模型runtime验证边界，
不是GPU数值精度或任务成功率测试。当前轻量QA环境无Torch，因此原生train/preprocess
完整CLI入口依赖检查会在import Torch处停止。未下载任务权重或基础大模型，
未执行GPU forward、训练、基准仿真或真机。
