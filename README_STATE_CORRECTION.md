# ReCogDrive：学生末期状态上的特权教师纠偏

## 1. 这次实验只回答什么

**教师适应学生实际访问的末期生成状态后，监督质量能否提高，并最终改善 OPD 学生？**

已观察到同样给 GT goal，教师在学生 z4 上 ADE≈0.621，而在 q(GT) 上≈0.119。该现象支持做状态适应实验，但尚不能证明它是学生分数下降的唯一原因。

本实现不添加学生 SFT，不换 backbone，不做多教师路由搜索，不改为除以极小方差的“严格 KL”。首轮只适应一位特权教师，再做原教师/适应后教师的两组 OPD 对照。

| 阶段 | 使用什么模型/数据 | 训练或验证目的 | 主要输出 |
|---|---|---|---|
| collect | 已有 OPD 后学生；训练/验证缓存 | 固定学生实际访问的末步状态 | 状态 `.pt` 与 manifest |
| finetune | 原特权教师；训练集学生状态 + 同场景标准 q(GT) | 学会从学生状态纠偏，同时保留原去噪能力 | 教师 `best.ckpt`、`last.ckpt`、日志 |
| validate | 微调前/后教师；同一份留出学生状态 | 配对检查纠偏是否改善 | 分步统计、逐样本结果、可选轨迹和 PDMS |
| opd original/adapted | 同一个原始 goal-free 学生初始化；分别使用两位教师 | 隔离教师状态适应对蒸馏的作用 | 两组学生 checkpoint |
| evaluate-student | 两组 OPD 学生；同一评测缓存/种子 | 检查最终驾驶效果 | ADE/FDE、可选 PDMS |

**收集状态的学生与 OPD 初始化学生是两个不同角色。** 默认前者是已有 OPD 后模型，后者是这次对照统一使用的原始 goal-free base。不是继续给采集学生做 SFT。

## 2. 文件与安装

新增：

- `navsim/agents/recogdrive/student_state_correction.py`：缓存读取、模型加载、采样、教师纠偏目标、单教师 OPD、PDMS 接口。
- `navsim/planning/script/run_student_state_correction.py`：五个阶段的 CLI，训练支持单机 torchrun/DDP。
- `scripts/training/student_state_correction.example.json`：路径占位和全部实验设置。
- `scripts/training/run_student_state_correction.sh`：统一启动入口。
- `tests/test_student_state_correction.py`：CPU 数值、状态协议及流程测试。
- `tests/smoke_correction_ddp.py`：两进程 CPU DDP 计算图测试。

修改：

- `navsim/agents/recogdrive/recogdrive_goalfree_kd_only_distill_trainer.py`：增加独立采样参数、按步 KD 输出梯度指标、非有限 loss 的可选报错；增加 `kl_proxy_loss` 指标，保留旧指标名和旧构造参数默认行为。未修改其他旧实验的配置/启动脚本。

完整代码包基于上传的 `recdrive-multi-opd-v1-gpt-small.zip` 合并上述文件。也可以从包中取出这些文件，按相同相对路径覆盖服务器仓库。已有代码发生过改动时，先对照包内 `STATE_CORRECTION_CHANGES.patch`；不要覆盖自己之后的修改。

运行环境使用原 ReCogDrive/NAVSIM 环境及依赖。本入口直接加载 action head 和已有 VLM hidden-state cache，**不加载 VLM 权重，也不重新提取特征**。小模型默认 hidden state 最后一维为 1536；必须与训练原 checkpoint 的 VLM/cache 一致。

## 3. 必须填写的配置

复制 `student_state_correction.example.json` 为自己的配置，再修改下列内容。`/FILL/` 都是待填写路径，不是有效默认路径。

| 字段 | 填写内容 |
|---|---|
| `collector_student.checkpoint` | 已有 OPD 后、确认存在末步退化的学生 checkpoint |
| `collector_student.kind` | 无 goal 为 `goal_free`；带预测 goal 为 `pred_goal`，不得用 goal-free 类强行丢弃预测头 |
| `teacher.checkpoint` | 一位原始特权教师 checkpoint |
| `teacher.goal_mode` | 与教师训练时一致：`adaln`、`cross` 或 `channel` |
| `teacher.goal_kwargs` | 与原结构一致的 goal encoder 参数；保留训练时的 goal 扰动参数 |
| `opd_student.checkpoint` | 两组 OPD 共同的原始 goal-free 初始化 checkpoint；不要默认填旧 SFT checkpoint |
| `train_indexes` / `val_indexes` | 现有 direct index JSON 列表，分割应按 log 且互不重叠；可只填写这位教师对应的数据范围 |
| `state_dir` | 新的空目录，保存固定学生状态；状态收集拒绝覆盖非空目录 |
| `output_dir` | 新实验输出目录；训练拒绝覆盖已有 `last.ckpt` |
| `metric_cache_path` | 可选，NAVSIM v1 PDMS metric cache 根目录；使用 `--pdms` 时必须填写 |
| `evaluation_indexes` | 可选，最终评测 direct index；空列表时使用 val_indexes，不代表自动用了 navtest |
| 两个学生的 `adaln_bound` | 必须符合各自 checkpoint 当时的行为；示例 OPD 学生为 8，未使用 soft bound 的 base 可设 0 |

模型加载接受 `agent.action_head.*`、`action_head.*`、相应 `module.*` 前缀及纯 planner checkpoint。只选 action head，不误加载 checkpoint 里的冻结教师。所有模型参数都必须有匹配权重；goal 结构、张量形状或时间表不符会报错，不会静默随机初始化。

如果采集模型是 pred-goal，示例增加：

```json
"kind": "pred_goal",
"goal_mode": "adaln",
"goal_kwargs": {
  "goal_sincos_dim": 128,
  "goal_hidden_dim": 1024,
  "goal_use_heading": false,
  "goal_predictor_hidden_dim": 512,
  "goal_predictor_dropout": 0.0
}
```

这只改变状态采集模型；首轮下游 OPD 仍固定 goal-free 学生。尽量优先用已经复现该问题的 goal-free OPD 学生采集；如果问题仅在 pred-goal 上确认，就如实记录这次训练用的是 pred-goal 状态，不能声称它与 goal-free 学生访问分布完全相同。

### direct index 格式

沿用原代码 `DirectIndexCacheDataset` 的两种格式。例如：

```json
{
  "builders": ["internvl_feature", "trajectory_target"],
  "samples": [
    {"token": "场景token", "path": "/cache/log_name/token"}
  ]
}
```

也支持 `cache_path` + `tokens: {token: relative_path}`。每个 token 文件夹需有 `internvl_feature.gz` 与 `trajectory_target.gz`，内容与现有缓存相同。log 按 token 文件夹的父目录名判断；目录布局不同，应先调整索引/分割规则。训练/验证 token 或 log 重叠会报错，不通过随机分帧掩盖泄漏。

## 4. 采样设置必须固定

示例 `sampler` 使用你现有 OPD 的随机采样设定：`min_sigma=0.04`、`noise_clip=5`、`deterministic=false`，用于**状态采集和两组 OPD**。这是保留实验条件，不是建议部署也加大噪声。

`evaluation_sampler` 默认 `min_sigma=0.0001`、`noise_clip=1`，用于最终学生评测和教师完整 rollout，符合原常规推理默认值。固定状态纠偏评测始终使用保存的 z4，无须重新采样。

如果已经修复 OPD 使其与部署一致，就在首次收集前把 `sampler` 也设置为 `0.0001 / 1`，并让两组 OPD 一致。**修改采样设置后必须重新收集状态，不要复用旧 manifest。** 清单会记录并检查设置。

`opd.loss_min_sigma=0.04` 仅用于 KD 代理损失权重，独立于采样下限。它不决定采样轨迹；末步仍使用有限权重，不除以接近零的真实方差。

采集保存每次 DiT 调用前的状态。五步时默认只保存 `step=4`（z4），其真实 timestep 通常为 0；z4 不是最后一次更新后的最终轨迹。需要同时适应 z3 时改 `collect.steps` 为 `[3,4]` 并重新收集。

## 5. 执行顺序

当前 general 首轮配置使用：

```text
CONFIG=scripts/training/student_state_correction.pre_goal_general.json
state_dir=/mnt/datasets/recdrive/20260513/state_correction_pre_goal_general_v1/student_states
output_dir=/mnt/datasets/recdrive/20260513/state_correction_pre_goal_general_v1/results
```

完成 collect 后，按以下顺序继续：

```bash
# 1. 微调 general 特权教师
bash scripts/training/run_student_state_correction.sh finetune

# 2. 配对验证原教师与适应后教师
bash scripts/training/run_student_state_correction.sh validate --export-trajectories

# 3. 计算固定 z4 PDMS 和完整教师 rollout PDMS
bash scripts/training/run_student_state_correction.sh validate \
  --pdms --full-rollout --export-trajectories

# 4. 生成 loss、梯度冲突、验证曲线和代表性轨迹图
bash scripts/training/run_student_state_correction.sh plot

# 5. 用原教师训练 goal-free 学生基线
bash scripts/training/run_student_state_correction.sh opd --arm original

# 6. 用微调后的教师训练对照学生
bash scripts/training/run_student_state_correction.sh opd --arm adapted

# 7. 评测两组学生
bash scripts/training/run_student_state_correction.sh evaluate-student --arm original --pdms
bash scripts/training/run_student_state_correction.sh evaluate-student --arm adapted --pdms
```

`finetune` 和 `opd` 使用 `GPUS=8` 的 torchrun；`validate`、`plot` 和 `evaluate-student` 是单进程。只有 paired validation 已导出轨迹后，`plot` 才会生成代表性轨迹图；训练曲线和梯度冲突图会直接生成。

先激活原项目环境，进入仓库根目录：

```bash
cp scripts/training/student_state_correction.example.json /你的路径/correction.json
# 编辑 JSON，填完 /FILL/ 路径及模型/数据设置。
export CONFIG=/你的路径/correction.json
export GPUS=8
# python/torchrun 不在 PATH 时设置 PYTHON_BIN、TORCHRUN_BIN。
```

### A. 收集状态：只推理，不训练学生

```bash
bash scripts/training/run_student_state_correction.sh collect
```

默认单进程、使用一张可见 GPU。固定种子按 token 派生，不依赖样本顺序。收集 train 和 val 两份状态；轻量状态文件只保存归一化 z4、step/timestep、学生最终轨迹、token 与种子。冻结的 hidden-state/history/status/GT 不重复写入状态文件，finetune/validate 时通过 manifest 中的原 cache 路径重新读取；因此正式实验期间必须保留原 feature cache 及其路径。

`seeds_per_scene=1`；`max_train_samples=0` 和 `max_val_samples=0` 表示全部。首次联调可设为 32/16，验证运行链路后换新目录收集正式数据。限量时用固定哈希顺序抽取，不按时间顺序截取。

### B. 教师微调：学生状态纠偏 + 原始去噪保持

```bash
bash scripts/training/run_student_state_correction.sh finetune
```

目标：

```text
L = correction_weight × MSE(x0_T(z4_student, GT_goal), GT_trajectory_normalized)
  + retention_weight × 原教师标准 q(GT) epsilon 回归损失
```

每个训练样本都计算一份纠偏损失和一份保持损失，相当于 1:1 的两类训练目标；不是随机一半样本不纠偏。两项数值尺度不同，示例权重 1/1 是起点，并不保证梯度贡献相等。

- 纠偏分支给干净 GT goal，关闭 goal mask/noise；输入真实缓存 z4，不再额外加噪。
- **只做一次教师前向，预测 epsilon 并转换为 x0**；不从 z4 再虚构多个去噪步骤。
- 保持分支沿用 `GoalCondDiffusionPlanner.forward`：随机训练 timestep、标准前向噪声、原 goal 扰动。
- 原教师仍是 epsilon 参数化；没有换成直接 x0 输出网络。回归目标用 x0，避免给非标准学生状态套用随机噪声标签。
- 损失在实际裁剪后的 x0 上计算，与你后续 OPD 取到的教师输出一致；如果严重饱和，裁剪可能阻断梯度，此时日志会暴露问题，不能把 loss 权重放大当作解决方案。
- 更新整个 planner 的可训练参数；固定 eta 不训练；没有 VLM 训练。
- FP32、小学习率、梯度裁剪；默认单卡 batch=4、5 epochs，只是初始预算，非已验证最优参数。

每个 epoch 用相同留出状态验证，`best.ckpt` 按最晚收集步骤的教师 ADE 选择，`last.ckpt` 是最后一轮。**best 不是按 PDMS 选出的“最优驾驶模型”**；请结合下一阶段 PDMS 和正常 rollout 结果判断。

### C. 配对验证教师

只计算几何指标：

```bash
bash scripts/training/run_student_state_correction.sh validate --export-trajectories
```

填写 metric cache 后，同时计算学生状态上 x0 的 PDMS，以及教师完整 rollout 的保持情况：

```bash
bash scripts/training/run_student_state_correction.sh validate \
  --pdms --full-rollout --export-trajectories
```

默认比较 `teacher.checkpoint` 与 `output_dir/finetune/best.ckpt`；可用 `--checkpoint /path/to/teacher.ckpt` 指定其他 epoch。

输出在 `paired_validation/`：

- `summary.json`：每一步分别统计 mean/median/P90、数量；`step=-1` 单独代表完整 rollout。
- `per_sample.json`：同一 token/seed/step 上两位教师的配对结果；可选物理坐标轨迹，便于现有 viz 读取。
- `provenance.json`：checkpoint SHA256、状态清单 SHA256、设置及是否启用 PDMS。

`x0_pdm_score` 是**教师在固定学生 z4 上单次输出的轨迹**经 NAVSIM simulator/scorer 后的 PDMS；`full_pdms` 是教师从初始噪声完成完整采样的 PDMS。两者不能混为一谈。

PDMS 接入使用仓库中的 NAVSIM v1 `pdm_score`、40×0.1s proposal sampling 和默认评分权重，分值通常为 0–1，展示 0–100 时统一乘 100。没有实现 v2 EPDMS 的新评分器；最终 EPDMS/NavHard 继续用服务器原有对应评测入口，加载这里导出的学生 checkpoint。开启 PDMS 时任何缺失 token metric cache/评分异常直接报错，不填零或悄悄丢样本。

### D. 两组单教师 OPD

```bash
bash scripts/training/run_student_state_correction.sh opd --arm original
bash scripts/training/run_student_state_correction.sh opd --arm adapted
```

两组都从 `opd_student.checkpoint` 开始，分别冻结原教师和 `finetune/best.ckpt`，训练数据、seed、采样、优化器、预算相同。两组必须使用相同 GPU 数量与 batch size。

保留原 goal-free KD trainer 的共享学生 DDIM 方差重建均值，以及有裁剪的 precision 加权均方差；**是 KL 代理损失，不是严格 KL**。五步平均；关闭外推、goal head、anchor、额外 jerk loss；一个教师覆盖选定训练集，不再进行语义 bucket 路由。学生状态每次按当前学生生成并 detach。

训练入口为独立 FP32 runner：与旧 Lightning 实验可能存在优化器、精度、缓存 padding 等差异。因此有效对照是**此入口中的 original vs adapted**，不能把其中一组直接与历史旧 runner 的单个分数当作唯一因果证据。

输出 `opd_original/`、`opd_adapted/`；每轮记录完整学生 rollout 的验证 ADE/FDE，best 暂按验证 ADE 选，last 按固定预算保存。正式驾驶分数比较优先同时报告两组相同预算的 last，以及相同选模规则的 best，不用测试集逐 checkpoint 挑分。

### E. 最终学生评测

```bash
bash scripts/training/run_student_state_correction.sh evaluate-student --arm original --pdms
bash scripts/training/run_student_state_correction.sh evaluate-student --arm adapted --pdms
# 比较相同预算的 last 时可加：
# --checkpoint /你的输出目录/opd_original/last.ckpt
```

这里按 `evaluation_sampler` 完整采样学生，逐 token 使用相同种子；默认评测 best，可覆盖 checkpoint。设置 `evaluation_indexes` 才能指定 navtest/NavHard 对应特征缓存；空列表只评测 val。最终测试集不参与教师微调和选模。输出位于 `evaluate_student_original/` 或 `evaluate_student_adapted/`。

新 checkpoint 的参数前缀是 `agent.action_head.*`，可由原仓库对应的普通学生/特权教师加载器读取。checkpoint 不含 Lightning optimizer/scheduler 状态；这是模型导出，**不能作为原 Lightning trainer 的断点续训 ckpt_path**。将它作为 agent 的初始化 checkpoint 使用；原评测入口也必须设置与训练一致的学生 adaLN bound 等非权重配置，不能假设仅加载权重就恢复了这些设置。

## 6. 指标怎么读

| 阶段/指标 | 含义 | 希望看到什么 |
|---|---|---|
| teacher `z4/ade_m` / `fde_m` | 教师在固定学生状态上的 GT 误差 | 留出样本均值和 P90 降低 |
| `correction_m` | denorm(x0) 与 denorm(z4) 的 XY 位移量 | 与误差下降一起看；不是越大越好 |
| `ade_gain_from_state_m` / `z4/ade_gain_m` | 纠偏前状态误差减去输出误差 | 正值表示修正有效；z4 是带噪输入，此项只是诊断 |
| `paired_ade_gain_m` / `ade_improved` | 微调前后同状态 ADE 改善量/比例 | 改善不是仅集中于少量样本 |
| `qgt_ade_m` / `baseline_qgt_ade_m` | 同一步、同一份标准 q(GT) 输入上的误差 | 正常去噪能力没有明显退化 |
| `x0_pdm_score` / `paired_pdms_gain` | 固定学生状态输出轨迹的驾驶分数 | 提升才说明驾驶层面纠偏有效 |
| `full_pdms` / `baseline_full_pdms` | 教师常规完整采样分数 | 原有规划能力保持 |
| `eps_abs_mean` / `clip_fraction` | 输出幅度、raw x0 被裁剪的比例 | 警惕输出剧增且大量饱和 |
| `corr_grad_eps_norm` | 单步纠偏 MSE 对教师 epsilon 输出的梯度范数，未乘总权重 | 检查低噪声梯度是否极弱；它不是参数梯度 |
| `branches/*_grad_norm` | correction/retention 对教师全部可训练参数的独立梯度范数，并同时记录乘 loss 权重后的范数 | 判断 `1/1` loss 权重是否真的带来同量级参数更新 |
| `decoder_branches/*_grad_norm` | 两分支对 action decoder 参数的独立梯度范数 | 检查冲突是否集中在轨迹解码核心 |
| `branches/gradient_cosine` / `decoder_branches/gradient_cosine` | 两分支参数梯度的余弦相似度 | 持续小于 0 表示更新方向冲突；接近 0 表示大体独立 |
| `branches/weighted_grad_norm_ratio` | 加权 correction 梯度范数除以加权 retention 梯度范数 | 远小于 1 表示 retention 主导，远大于 1 表示 correction 主导 |
| `branches/gradient_conflict_fraction` | 两分支共同非零的梯度元素中符号相反的比例 | 与 cosine 一起看，不单独作为调权依据 |
| `decoder_total_grad_norm` | 完整训练 loss 对 action decoder 参数的梯度，裁剪前 | 教师阶段是纠偏+保持合计，不能误称纠偏分支单独贡献 |
| OPD `kl_proxy_loss` | 有限权重的共享均值 KD 代理损失 | 只反映拟合，不能独立证明驾驶改善 |
| OPD `kd_grad_x0_step_i` | 加权且除以步骤数的第 i 步 KD 对学生 x0 输出梯度 | 查看末步监督是否有学习信号；非 epsilon/参数梯度 |
| OPD 按步 ADE/FDE、x0 gap、`kl_step_i_fraction` | 各步预测质量与 loss 贡献 | 教师改善能否被学生吸收、是否有步骤独占损失 |
| 最终学生 PDMS / EPDMS / NavHard | 最终任务表现 | 需要真实评测，不从 ADE 改善推断必然涨分 |

`train.jsonl` 每 `diagnostic_interval` 个优化步写一次，同时写入 `finetune/tensorboard/`。分支冲突指标是 rank0 本地 batch 的未同步诊断梯度（在 DDP `no_sync` 下单独计算，不参与参数更新）；实际 optimizer 使用的总 loss 参数梯度仍由 DDP 平均。验证阶段由所有 rank 按不补齐的分片共同遍历留出集，每个 chunk 用 `all_gather_object` 汇总，rank 0 负责写报告和 checkpoint，避免其他 rank 在 rank 0 单独读完整验证集时等待 NCCL barrier 超时。`validation_samples=0` 表示全部，联调时可以减小，但两组必须一致。

训练后生成 loss、验证指标和梯度冲突总览图：

```bash
bash scripts/training/run_student_state_correction.sh plot
```

执行带 `--export-trajectories` 的 paired validation 后再次运行同一命令，还会生成改善最大、退化最大和中位样本的少量配对轨迹图及 paired ADE gain 分布；不为全部验证样本逐张出图。

所有统计按 sample/seed 记录；同一个场景多种子不是独立场景，不能直接用总记录数充当独立统计样本量。实际比较建议同时查看失败场景列表，尤其原长尾场景。

## 7. 预期结果与后续判断（假设，不是已跑出的结果）

| 观察 | 解释和下一步 |
|---|---|
| 教师 z4 ADE/PDMS 改善，正常能力保持，学生分数提高 | 支持“适应学生生成状态能提供更有效的 OPD 监督” |
| 教师 z4 改善，学生无提升 | 状态适应有效但不足；后续再研究学生吸收能力、末步权重或特权信息可实现性 |
| 教师 ADE 改善但 x0 PDMS 下降 | 更贴 GT 不等于更好驾驶，不能将其当有效监督修复 |
| 教师纠偏 loss 不降，epsilon 梯度很弱/裁剪严重 | 优先检查参数化与饱和；不要直接除以极小方差 |
| 教师 z4 改善但 q(GT)/完整 PDMS 明显退化 | 遗忘原能力；调整保持目标、学习率或训练时长后再决定是否蒸馏 |
| 只在收集学生的状态改善，新学生仍失败 | 固定状态覆盖不足；确认首轮结果后再考虑多 checkpoint 状态或刷新缓存，不在首轮同时增加变量 |

无需预设微调后一定接近 0.119，也不要把纠偏量增大或 KD 变小当作成功。

## 8. 实现边界与检查

- 已通过 14 项 CPU 数值/协议测试，覆盖状态索引、采样一致性、末步共享均值、分离 loss/采样方差下限、教师纠偏反传、OPD 教师冻结、checkpoint 导出和训练/验证分割等。
- 已通过两进程 CPU DDP smoke，包括 correction/retention 参数梯度冲突诊断与随后总 loss backward；它使用小型可微 surrogate planner，检查计算图/同步，不是实际 ReCogDrive 性能测试。
- 未在此环境加载你的真实 checkpoint、访问真实 cache、运行 GPU 训练或实际 PDMS；依赖兼容与分数需要服务器验证。
- 每个样本单独前向后，在本地 batch 内平均 loss，避免 variable-length hidden-state padding 造成阶段差异；吞吐可能低于原大 batch 实现。
- FP32，无 AMP，无自动断点续训，无多节点 shell 封装；Python 入口可由已有 torchrun 多节点参数启动，但未验证多节点环境。
- 状态缓存不重复保存冻结的 hidden states；训练和验证会按 manifest 路径读取原 feature cache，源 cache 不可删除或静默替换。训练和测试 checkpoint/pickle 必须来自可信来源。
- fixed-state 验证同时依赖 train/val manifest、相对 `.pt` 文件结构和 manifest 记录的原 cache 路径。正式比较应保存配置和清单，不手工替换缓存内容。

运行测试：

```bash
python -m unittest discover -s tests -p test_student_state_correction.py -v
torchrun --standalone --nproc_per_node=2 tests/smoke_correction_ddp.py
```
