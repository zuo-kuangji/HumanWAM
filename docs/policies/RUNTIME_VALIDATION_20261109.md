# 实际运行验证（2026-11-09）

这些检查验证加载、数值输出和服务协议，不是任务成功率测试，也没有连接机器人执行动作。
NTU已有八卡训练保持运行；GPU推理使用G1服务器空闲的RTX4090。

| Policy | 实际完成的检查 | 结果和限制 |
|---|---|---|
| ImageWAM Δ-flow | 真实靠垫H16、垃圾桶H16、衣服legacyH40权重，实际录制双摄图像／当前state44／严格过去history；每组连续3次推理 | 全部输出有限的16×76或40×76；相同seed逐值相等，不同seed有变化 |
| ImageWAM HTTP | 真实靠垫权重的localhost服务，health／predict／reset | predict有限16×76，与直接调用same-seed最大差0；future_state被400拒绝；临时服务已停止 |
| π₀.₅ A2A | 真实LIBERO第3560步JAX LoRA checkpoint、固定OpenPI+A2A patch、新wrapper，CPU限制4核 | 连续两次输出有限10×7；输入为合成RGB/state/history，未测试机器人质量 |
| GR00T N1.7 | 官方真实两分片checkpoint、固定7aa源码、新wrapper、三处权重与safetensors逐值核对、原生ZeroMQ往返 | 9组动作均为有限float32 [1,40,D]；get_action／get_modality_config／reset通过；使用基座实际支持的G1相对EEF profile |
| DiT4DiT | 官方LIBERO配置与统计、原生入口、CPU接口及发布配置回归 | 修复发布配置缺少image_size时的加载问题；没有实际GPU推理，见下文权限与环境条件 |
| Psi-Zero | 官方完整run的真实metadata、原生API、CPU接口回归 | 确认action80／raw state45／H30／ckpt40000，修复state43会丢失neck状态的配置；尚未加载11.22GB实际权重 |
| Fast-WAM | 官方固定源码、发布权重metadata和小统计、真实Hydra compose、原生API签名、CPU边界 | LIBERO A7/state8/H32与RoboTwin A14/state14/H32；未下载大权重或运行GPU推理 |
| ImageWAM joint41/47 | 新数据映射、原生FM/IC、H16、stop-gradient、state隔离、checkpoint绑定、SONIC v1离线桥 | 未训练关节模型，未运行SONIC encoder/decoder闭环或真机 |

## 实际发现并处理的问题

### ImageWAM 的环境冲突

G1原ImageWAM环境的用户site安装了`huggingface-hub1.5.0`，与`transformers4.56.1`
要求冲突。验证使用独立overlay环境固定`huggingface-hub0.36.2`，共享已有Torch库，
没有修改原环境。可复现的推荐依赖已在`requirements/wholebody-cu128.txt`独立列出。
本次实际推理环境是Python3.10、Torch2.7.1+cu126、Transformers4.56.1、Hub0.36.2。

### GR00T 的 embodiment 与离线资源

缓存的官方N1.7基座包含`real_g1_relative_eef_relative_joints`等profile，
不包含`unitree_g1_sonic`。新检查会在加载模型前明确拒绝错误tag，不会悄悄换成另一种机器人。
因此本次通过的是N1.7基座推理与协议，不代表现有SONIC78任务权重已经完成验证。

输入图像来自实际录像，pose输入使用checkpoint统计中的mean，明确属于合成pose。
离线tokenizer仍会尝试查询HF模型metadata，验证中建立独立checkpoint view，仅将两份
JSON中的backbone资源位置改为本地Cosmos缓存。权重与统计通过symlink读取原件，未修改。
修改前后SHA、原始权重参数核对及实际runtime版本均已记录。

精确源码是`7aa0bca50f1097f8b17f5c274c98264b72a1419f`，实际环境为Python3.10／Torch2.7.1。
它与上游声明的Python3.12／Torch2.9不同；本次检查不能扩展为所有配置的版本兼容承诺。

### DiT4DiT 的访问与环境条件

官方LIBERO模型约21.28GB，原生加载还需约21.23GB的Cosmos-Predict2.5-2B。
现有账号读取Cosmos的小型`model_index.json`返回`GatedRepoError / HTTP403`，
没有接受新条款或绕过访问限制，也没有下载大权重。
此外，当前G1的diffusers0.35.1没有`Cosmos2_5_PredictBasePipeline`，
需要按官方要求另外建立Cosmos环境，不能覆盖GR00T环境。

官方LIBERO实际配置为action8／state16／H8，没有image_size。适配器现仅在checkpoint
明确提供尺寸时resize，否则保留输入分辨率，防止因该字段缺失而加载失败。

### Psi-Zero 的实际发布配置

官方SONIC multi-task发布run是action80／state45／H30。旧的43维示例会遗漏neck2，
现在按实际metadata检查并拒绝静默丢弃非零状态维度。
只获取了36,340字节配置与统计；模型权重11.22GB尚未获取。
NTU现有ImageWAM环境无法导入原生Psi服务器所需的fastapi等依赖，
所以需要独立Psi环境，不能把CPU适配器检查写成完整模型推理成功。

## 代码检查

原有五个policy家族的轻量测试、原生CLI参数检查、原生ZeroMQ序列化与错误边界检查均保留。
ImageWAM原生H16／legacyH40的65项CPU测试已通过。
新增实际发布metadata fixture用于防止把伪造的小型fixture当作真实checkpoint配置。
所有临时预测服务在验证后关闭，现有模型、数据和训练作业没有被覆盖。

## Fast-WAM 与关节路线增量检查

当前为6个policy家族、7个注册入口，ImageWAM的motion76与joint41/47分开注册。
轻量package的111项CPU测试通过，含SONIC桥接29项和Fast-WAM的Hydra追加参数绕过输出目录保护的回归检查。
无上游checkout的CI会跳过依赖其源码的检查，实际固定Fast-WAM checkout上的Hydra与API检查已单独通过。

G1独立CPU环境的最终检查为原生模型55项、训练CLI23项、joint dataset14项，
另有未改动的旧dataset22项，共114项通过。设置`CUDA_VISIBLE_DEVICES=''`，确认CUDA不可见，
没有构造完整FLUX.2大模型；原生训练/损失测试通过AST提取真实代码配合小型模块执行。
它们覆盖41/47的FM与IC、无效姿态后6维零梯度而前41维仍训练、state不进入视觉、
动作梯度隔离、H16形状、114共享初始化，以及不同schema/任务统计/instruction的checkpoint拒绝。
这不能代替完整模型的GPU显存、吞吐或任务表现验证。

真实靠垫数据78条、65,963帧完成41/47重新统计与RGB解码检查，原文件SHA保持不变。
47D姿态按记录器有效性标记识别出63,765帧有效、2,198帧无效；无效姿态只影响6维监督，
不删除身体/手部标签或episode。缺少有效性策略的早期47D统计不再被接受。

SONIC桥接完成官方packer逐字节比对和官方unpacker往返，
H16/stride2的一秒参考可重采样为51帧50Hz；连续30FPS的H16覆盖不足会拒绝。
Inspire12单独导出，不复用Dex3命令字段。测试不发送网络消息，也没有启动机器人执行。
