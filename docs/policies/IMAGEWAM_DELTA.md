# ImageWAM Delta-flow

统一入口使用 `humanwam.backends.imagewam_delta`。新全身配方以 `src/imagewam/wholebody_delta.py`
和 `scripts/wholebody/` 为原生实现，不重新实现一套 FM/IC 计算。

新训练默认 history16、action16、future-image offset16；显式 `--horizon40` 才生成旧40步配置。
state44仅进入动作专家，视觉/文本不接收数值state。视觉source为
`0.5*z_current + sqrt(0.75)*noise`，动作source为归一化历史加`0.5*noise`，76维全部加噪。
视觉/动作各保留原生FM+IC，动作到视觉stop-gradient保持开启。

动作76是motion64和左右各6维实测Inspire状态，不能直接当作SONIC原生78D手部控制指令。
输出手部预测可能超出物理范围，机器人执行端需沿用经过验证的映射、限位和motion decoder。

## 新训练

在独立的ImageWAM/FLUX.2环境中安装根目录源码及轻量HumanWAM包。
依赖见`requirements/wholebody-cu128.txt`，不能用GR00T环境覆盖它。
完整数据、文本缓存和配置生成步骤见[全身指南](../G1_WHOLEBODY_DELTA.md)。

先用原生`imagewam.datasets.wholebody`生成任务统计，再用`cache_text.py`生成纯instruction缓存，
最后用`build_config.py --horizon16`按实际加载器生成正式与5步smoke配置。
smoke结束后正式训练重新从基座初始化；原生train入口会检查smoke证明、配置hash和horizon。

```bash
export HUMANWAM_ROOT=/path/to/HumanWAM
export IMAGEWAM_PYTHON=/path/to/imagewam-env/bin/python
export IMAGEWAM_NATIVE_CONFIG=/path/to/new-run/configs/production.yaml
humanwam train --config configs/humanwam/imagewam-train.json --dry-run
humanwam train --config configs/humanwam/imagewam-train.json
```

命令仅接受构建器生成且带`build_receipt.json`的原生配置。
新开训请按全身指南先执行对应smoke配置，不能拿未验证的模板直接代替production。

## 新模型推理

`imagewam-h16.json`需要完整的训练配置、checkpoint和该任务的Qwen缓存。
训练配置内的base/AE/action initializer、stats路径也必须指向当前机器真实文件。
输入NPZ包括`ego_rgb`、`chest_rgb`、`current_state[44]`、`past_actions[N,76]`。
图像为当前uint8 RGB，历史是严格过去的原始动作值，不能把尚未执行的整段预测作为历史。
初始时没有历史用空`[0,76]`，native source使用零归一化history后再加噪。

## 已发布模型

- `imagewam-trash-release.json`使用HF发布包自带的、manifest核验过的`TrashPolicy`。
- 靠垫使用同一配置，将`release_kind`改成`pillow`，仍是H16。
- `imagewam-clothes-legacy40.json`对应`ZUO66/imagewam-sonic-v11-deltaflow`的
  `inspire76-p25-ep025/inspire76-p25/epoch-025`完整快照，保留其40步和parent归一化。
  它使用旧release factory严格加载，不给旧checkpoint换成新wholebody格式。
- 微波炉/玩偶的历史task-specific发布格式继续使用各自发布包原生脚本，不能自动改format字段冒充新wholebody权重。

`humanwam serve`对该backend启动本仓库串行JSON/NPZ HTTP预测服务，默认localhost8090。
`POST /predict`内容为`{"observation_npz":"<base64 NPZ>","metadata":{}}`；
`GET /health`、`POST /reset`提供状态和重置。它不执行机器人动作，也不伪装成SONIC78 ZeroMQ协议。

本文维护了代码与输入边界，新增统一wrapper尚未做真实GPU权重/机器人回放验证。
原生H16/H40路由、损失、梯度、数据、训练CLI由CPU测试覆盖。
