# 严格在线联合 GRPO 训练手册

本文是当前正式训练流程。它使用固定 ICRP registry/ODE 作为外部 verifier，同时在线训练 Qwen LoRA 和 MLP gate。不会读取预先保存的 operator reward，也不会把隐藏真值放入 prompt。

## 1. 目录约定

```text
config/icrp_reviewed_models.json       人工审核的医学配置
data/icrp_model_registry_v1/           固定、可执行的 ICRP registry
data/icrp_knowledge_base_v2/           页面可追溯的检索索引
data/synthetic_bioassay_icrp_v1/       train/validation/test 合成病例
models/Qwen3-4b/                       冻结的 Qwen3-4B 基座
models/archive/smoke/                  已完成的 smoke 产物，仅供追溯
models/formal/                         正式训练输出
src/train_online_grpo.py                唯一正式在线联合训练入口
```

不要把 `models/archive/smoke/` 中的 adapter 或 gate 用作正式 checkpoint。`src/train_joint_grpo.py` 和 `src/train_qwen_grpo.py` 是旧的离线/counterfactual baseline 或 helper，不是本流程的正式入口。

## 2. 训练语义

每个训练状态按以下顺序处理：

1. 用确定性 profile 对病例执行固定 ICRP ODE，得到当前数值状态。
2. MLP gate 输出 `SKIP_LLM`/`CALL_LLM` 概率，并按探索概率采样本步 active action。
3. 执行 deterministic skip 分支并实时得到 `skip_reward`。
4. 从当前 Qwen LoRA 采样一个 GRPO group；每个回答都经过 JSON、算子白名单、注册模型、参数边界和受保护字段审计。
5. 每个合法 Qwen proposal 都重新调用固定 ODE；非法 proposal 得到硬负奖励，不进入 ODE。
6. 用实时 `call_reward - skip_reward` 更新 gate；只有 active action 是 `CALL_LLM` 时才更新 Qwen LoRA，组内优势使用在线 reward 计算。
7. registry、compartment、transfer、rate、衰变常数、measurement mapping 和 ODE 代码始终冻结。
8. 每一步写入 `samples.jsonl`，最终保存 Qwen adapter 和 gate checkpoint。

训练阶段为了获得 gate 的 counterfactual target，即使 active action 是 `SKIP_LLM`，也会评估 Qwen call 分支；这保证 gate 的监督来自同一状态上的真实 ODE 对照，而不是离线标签。

## 3. 环境与预检

在项目根目录执行。`PYTHON` 可以替换为你的环境 Python；当前验证环境为 `/root/miniconda3/envs/toolkit/bin/python3`。

```bash
export PYTHON=/root/miniconda3/envs/toolkit/bin/python3
export MODEL=models/Qwen3-4b
export FORMAL=models/formal/online_joint_grpo_icrp_v1

$PYTHON -c "import torch, transformers, scipy; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
nvidia-smi
$PYTHON -B -c "import ast, pathlib; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in pathlib.Path('src').glob('*.py')]; print('source syntax ok')"
```

如果人工审核配置发生变化，先重建并验证 registry：

```bash
$PYTHON src/build_icrp_model_registry.py \
  --config config/icrp_reviewed_models.json
$PYTHON src/validate_synthetic_bioassay_dataset.py \
  --dataset-dir data/synthetic_bioassay_icrp_v1 \
  --registry-dir data/icrp_model_registry_v1
```

正式训练前必须确认：registry `require_verified=True` 可加载，数据集的 train/validation/test 都存在，GPU 没有其他训练进程，显存至少留出约 20 GB 给 Qwen3-4B。

## 4. 正式训练命令

下面命令使用全部 210 个 train 病例循环训练 2000 个在线状态，Qwen3-4B、LoRA rank 8、双 rollout group，适合 24 GB RTX 4090 起步：

```bash
mkdir -p "$FORMAL"
$PYTHON -u src/train_online_grpo.py \
  --dataset-dir data/synthetic_bioassay_icrp_v1 \
  --registry-dir data/icrp_model_registry_v1 \
  --icrp-knowledge-dir data/icrp_knowledge_base_v2 \
  --split train \
  --cases 210 \
  --model-path "$MODEL" \
  --output-dir "$FORMAL" \
  --steps 2000 \
  --group-size 2 \
  --max-length 768 \
  --max-new-tokens 256 \
  --forward-budget 48 \
  --gate-learning-rate 1e-3 \
  --learning-rate 1e-6 \
  --gate-temperature 0.05 \
  --gate-exploration 0.20 \
  --temperature 0.7 \
  --top-p 0.9 \
  --rank 8 \
  --alpha 16 \
  --seed 17 \
  2>&1 | tee "$FORMAL/train.log"
```

不要同时运行 `train_joint_grpo.py`、`train_qwen_grpo.py` 或第二个 Qwen 进程。若显存不足，按顺序降低 `--group-size 1`、`--max-length 512`、`--max-new-tokens 192`；不要修改 ODE、registry 或 auditor。

如果希望先做短验证，只改变 `--steps` 和 `--cases`，并使用新的输出目录，例如 `models/formal/online_joint_grpo_check`；不要覆盖正式目录。

## 5. 运行监控

另开终端：

```bash
tail -f "$FORMAL/samples.txt"
nvidia-smi
```

统计 JSONL：

```bash
$PYTHON - <<'PY'
import json, os
p=os.path.join('models','formal','online_joint_grpo_icrp_v1','samples.jsonl')
rows=[json.loads(x) for x in open(p, encoding='utf-8') if x.strip()]
if rows:
    print('steps=', len(rows))
    print('mean_call_reward=', sum(x['mean_reward'] for x in rows)/len(rows))
    print('mean_skip_reward=', sum(x['skip_reward'] for x in rows)/len(rows))
    print('valid_fraction=', sum(x['valid_output_fraction'] for x in rows)/len(rows))
    print('active_call_rate=', sum(x['gate_action']=='CALL_LLM' for x in rows)/len(rows))
    print('mean_gate_call_probability=', sum(x['gate_call_probability'] for x in rows)/len(rows))
    print('mean_forward_calls=', sum(x['forward_predict_calls'] for x in rows)/len(rows))
PY
```

重点检查：`valid_fraction` 不应长期为 0；`mean_call_reward` 应与 `mean_skip_reward` 有可解释差异；`active_call_rate` 应受到 gate 概率和探索下限共同影响；forward 次数不能失控；日志中不能出现 OOM、NaN 或未捕获异常。

## 6. 正式输出

训练完成后，`$FORMAL` 应至少包含：

```text
adapter.pt       Qwen LoRA adapter
llm_gate.pt      MLP gate checkpoint
metadata.json    参数、算法、gate history 和训练摘要
samples.jsonl    每步完整在线 rollout、审计和 reward
samples.txt      便于 tail 的每步摘要
train.log        终端输出
```

`adapter.pt` 与 `llm_gate.pt` 必须成对使用；不能把一个正式 adapter 和另一个目录的 gate 混用。

## 7. 只读验证集/测试集评估

训练完成后，不要用 validation/test 更新任何参数。先用一个 validation case 验证 checkpoint 链路：

```bash
$PYTHON src/run_ode_dataset_pipeline.py \
  --dataset-dir data/synthetic_bioassay_icrp_v1 \
  --registry-dir data/icrp_model_registry_v1 \
  --icrp-knowledge-dir data/icrp_knowledge_base_v2 \
  --case-id ode-synthetic-000210 \
  --split validation \
  --planner mlp_llm_operator \
  --model-path "$MODEL" \
  --llm-gate-path "$FORMAL/llm_gate.pt" \
  --llm-adapter-path "$FORMAL/adapter.pt" \
  --rounds 2 \
  --output "$FORMAL/validation_smoke.json"
```

再对 test 做只读批量评估。不要把输出目录放回训练输入目录：

```bash
mkdir -p "$FORMAL/test_results"
for f in data/synthetic_bioassay_icrp_v1/inputs/test/*.json; do
  case_id=$(basename "$f" .json)
  $PYTHON src/run_ode_dataset_pipeline.py \
    --dataset-dir data/synthetic_bioassay_icrp_v1 \
    --registry-dir data/icrp_model_registry_v1 \
    --icrp-knowledge-dir data/icrp_knowledge_base_v2 \
    --case-id "$case_id" --split test \
    --planner mlp_llm_operator --model-path "$MODEL" \
    --llm-gate-path "$FORMAL/llm_gate.pt" \
    --llm-adapter-path "$FORMAL/adapter.pt" \
    --rounds 2 \
    --output "$FORMAL/test_results/$case_id.json" || exit 1
done
```

最终报告至少包含 objective/weighted loss、合成真值上的 intake 和 intake-time error、ODE forward 次数、LLM 调用率、合法 proposal 率、预算违规率和 STOP 决策统计。测试集只读，不参与训练。

## 8. 失败处理

- `No JSON proposal` 或 invalid proposal：先看 `max-new-tokens` 和 `valid_fraction`，不要绕过 auditor。
- CUDA OOM：确认没有其他 GPU 进程，再按第 4 节降低上下文和 group size。
- registry/模型校验失败：重新执行 registry 构建和数据验证，不要手工编辑 CSV。
- 训练中断：保留该输出目录和 `train.log`，不要把不完整目录冒充正式 checkpoint；重新使用新的输出目录启动。
