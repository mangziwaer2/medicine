# ICRP 固定生物动力学模型上的 LLM 优化

本项目是一个用于理论验证的内部污染生物测量反演工程。它研究的问题是：在生物动力学前向模型已经由 ICRP 文献固定、不能被语言模型改写的前提下，MLP 和 Qwen 是否能够学习“何时调用语言模型、下一步使用什么数值优化算子、给数值优化器提供哪些候选点”，从而减少无效 ODE 调用并改善摄入量/摄入时间反演。

项目不是临床剂量评估系统，也不声称已经完成临床验证。

## 当前唯一主线

### 目录交接约定

```text
config/icrp_reviewed_models.json   人工审核的 ICRP 模型输入（唯一医学配置入口）
data/icrp_model_registry_v1/       由配置生成的可执行 registry
data/icrp_knowledge_base_v2/       ICRP 文档检索索引
data/synthetic_bioassay_icrp_v1/   仅用于理论验证的合成病例
src/run_ode_dataset_pipeline.py    正式推理入口
src/train_online_grpo.py            严格在线 verifier-in-the-loop 联合训练入口
src/train_qwen_grpo.py             联合训练复用的 GRPO helper，不作为单独主线
docs/cloud_training.md              唯一训练与云端交接文档
models/和 temp/                     本地/云端生成的模型、数据和日志，不参与代码版本
```

```text
病例 JSON
  -> 严格审核的 ICRP 模型注册表
  -> ICRP 知识库检索（只提供来源上下文）
  -> profile 初始数值探测
  -> MLP gate: SKIP_LLM / CALL_LLM
  -> Qwen 选择白名单优化算子、注册 model_id 子集和有界候选点
  -> 计划审计（模型、边界、观测映射、预算）
  -> 固定 ICRP compartment ODE 前向计算
  -> L-BFGS-B / differential evolution / profile likelihood 等数值优化
  -> 摄入量、摄入时间、残差、调用成本和审计轨迹
```

Qwen 永远不能输出或修改 compartment、transfer、rate、physical decay、measurement mapping 或 ODE 代码。数值反演只优化 `log10_intake` 和 `intake_time_d`；前向动力学的所有结构、速率和观测映射只来自 `data/icrp_model_registry_v1`。

## 输入、输出与职责边界

病例输入由 `case_id`、受试者元数据、时间原点、候选摄入途径和一个或多个核素观测组成。每个核素只携带已注册的 `candidate_model_ids`；每条观测至少包含 `time_d`、`type`、`value_bq`，并可包含 `sigma_bq`、`detection_limit_bq` 和 `is_censored`。隐藏真值只存在于合成数据的 `labels.jsonl`，不会进入 planner prompt。

Qwen 的输出是受限 JSON 策略：一个白名单 operator、注册 model_id 子集、可选的有界候选点以及短理由。审计器会丢弃未知模型、越界点、重复点、受保护字段和预算违规动作。数值执行器最终输出每个核素的 `intake_bq_estimate`、`intake_time_d_estimate`、预测值、残差、objective、ODE 调用次数、所用算子和完整审计轨迹。

MLP 只决定 `SKIP_LLM`/`CALL_LLM`：前者执行确定性 fallback 算子并继续数值优化，后者调用 Qwen；只有收敛或不可辨识安全规则才能停止。Qwen 只决定优化策略；SciPy 优化器求连续变量；ICRP registry 是唯一前向模型来源。四者的权限互不重叠。

## ICRP 模型注册表

当前注册表由 `config/icrp_reviewed_models.json` 和 `src/build_icrp_model_registry.py` 生成，来源是仓库中保留的 ICRP Publication 137 PDF。JSON 配置文件是专业审核边界：专业人员只需审核其中的模型、区室、转移率、衰变常数、测量映射和来源页码；代码只负责结构校验、CSV 规范化和 ODE 执行，不判断医学内容是否正确。

- `ICRP137_I131_systemic_reference_worker`：Publication 137 Table 5.4，系统性碘参考工作人员模型。
- `ICRP137_Cs137_systemic_reference_worker`：Publication 137 Table 6.3，系统性铯参考工作人员模型。

每个模型的 `models.csv` 都必须包含 `meta_verification_status=verified_icrp_transcription`、`meta_registry_kind=icrp_verified`、Publication、页码和表号。`ODEModelRegistry.from_csv_dir(..., require_verified=True)` 会拒绝缺少这些字段的模型，也会拒绝 `demo`/`synthetic` 模型。

需要区分两层：compartment 拓扑、转移率和物理衰变常数来自 ICRP 137；`T-body`、`Thyroid-Bioa`、24 小时排泄量到 ODE 状态/通量的映射仍是本项目冻结的工程测量适配器，不是新的 ICRP 剂量系数，也不由 Qwen 学习或修改。专业人员可以在 JSON 配置中审核或替换这些映射。

当前反演量的语义是“进入 ICRP 血液/血浆系统区室的急性 activity 输入”，不是尚未实现的吸入或摄入途径外部 intake。这样可以直接使用 Publication 137 的系统性模型而不伪造 HATM 吸收参数。以后增加 route-specific 模型时，必须新增独立的 ICRP 来源注册条目，不能让 Qwen 自由构造。

## ICRP 知识库

`data/icrp_knowledge_base_v2` 是由仓库中的 9 份 ICRP PDF 建立的可追溯 JSONL 索引，包含 documents、chunks、sections、entities、mechanism_facts、parameter_candidates、model_templates 和 knowledge_cards。`data/icrp_knowledge_base_v1` 仅作为 v2 的原始文本构建索引保留，运行时默认检索 v2。检索器是 `src/icrp_knowledge.py` 的可复现词法检索器，返回 publication、page、chunk_id、official_url 和短文本片段。

知识库的作用是给 Qwen 提供“为什么这个模型/算子可能合理”的来源上下文；它不是第二个 ODE 参数源。自动抽取的数值默认 `needs_review`，不得直接注入前向模型。真正可执行的 transfer 表只在已审核注册表中出现。

## 专业审核配置与核素衰变数据

所有需要专业人员确认的医学内容集中在 `config/icrp_reviewed_models.json`。配置中包括区室、转移率、物理衰变常数、测量映射、摄入语义、来源出版物、页码、表号和 SHA-256。修改配置后重新执行：

```powershell
E:\Miniforge\envs\medicine\python.exe src\build_icrp_model_registry.py `
  --config config\icrp_reviewed_models.json
```

代码只做 JSON 结构检查、CSV 规范化和数值执行，不判断专业人员填写的内容是否医学正确。I-131、Cs-137 的物理衰变来源字段指向 ICRP Publication 107；如需更高精度，专业人员可使用 ICRP 107、IAEA LiveChart 或 NNDC NuDat 复核后直接修改配置。

NNDC NuDat 是美国 Brookhaven National Laboratory 的核素数据库：

- [NuDat 3](https://www.nndc.bnl.gov/nudat3/)
- [IAEA LiveChart](https://www-nds.iaea.org/relnsd/vcharthtml/VChartHTML.html)
- [ICRP Publication 107](https://www.icrp.org/publication.asp?id=ICRP%20Publication%20107)

使用 NuDat 时，在核素搜索框输入 `I-131` 或 `Cs-137`，查看 Nuclear Levels、Decay Radiation、Half-life 等条目。项目当前只需要核对半衰期并计算 `lambda = ln(2) / half_life`；NuDat 的能谱、分支比和辐射线数据暂不进入当前生物动力学 ODE。NuDat 是交叉核对工具，不是本项目的 compartment 或 transfer-rate 来源。

## 代码入口

- `src/run_ode_dataset_pipeline.py`：正式入口，默认使用 MLP gate + Qwen operator planner。
- `src/debug_pipeline.py`：可读的 smoke/debug 入口，默认使用 deterministic planner，不加载 Qwen。
- `config/icrp_reviewed_models.json`：专业人员审核的 ICRP 模型配置，包含区室、转移率、衰变常数、测量映射和来源。
- `src/build_icrp_model_registry.py`：将审核 JSON 转换为执行所需的 CSV registry。
- `src/compartment_ode.py`：固定线性 compartment ODE 和严格注册表加载器。
- `src/multi_nuclide_llm_optimizer.py`：病例 schema、ICRP 检索上下文、数值反演、算子审计和外层循环。
- `src/train_llm_gate.py`：训练 MLP 的二分类 gate。
- `src/train_qwen_lora.py`：监督式 operator SFT LoRA baseline。
- `src/train_qwen_grpo.py`：被联合训练器复用的 GRPO 编码、采样和 reward helper；云端不要单独启动它。
- `src/train_online_grpo.py`：当前唯一的严格在线 MLP gate + Qwen LoRA 联合训练入口；每个 proposal 都实时送入固定 ODE verifier。
- `src/train_joint_grpo.py`：旧的离线 counterfactual 联合训练 baseline，不用于当前正式主线。
- `src/build_llm_preference_dataset.py`：固定数值状态下枚举安全算子，建立 counterfactual reward 数据。
- `src/evaluate_operator_preferences.py`：检查 preference reward、regret 与 forward-budget 统计。
- `src/normalize_case_time_origin.py`：将无可靠暴露原点的结构化病例统一到明确的时间坐标。

## 快速运行

需要使用包含 `numpy/scipy/torch/transformers` 的 Python 环境。Windows 本地示例：

```powershell
E:\Miniforge\envs\medicine\python.exe src\debug_pipeline.py `
  --case-file data\synthetic_bioassay_icrp_v1\inputs\test\ode-synthetic-000255.json `
  --planner heuristic `
  --rounds 2 `
  --output temp\debug_pipeline_result.json
```

正式 Qwen 主线：

```powershell
E:\Miniforge\envs\medicine\python.exe src\run_ode_dataset_pipeline.py `
  --case-id ode-synthetic-000255 `
  --split test `
  --planner mlp_llm_operator `
  --rounds 2 `
  --output temp\mainline_result.json
```

无 Qwen 的可比基线：

```powershell
E:\Miniforge\envs\medicine\python.exe src\run_ode_dataset_pipeline.py `
  --case-id ode-synthetic-000255 `
  --split test `
  --planner heuristic `
  --rounds 2
```

正式入口会同时检查 ICRP registry 和 ICRP knowledge base；缺失或未审核模型会直接失败，不会回退到内置 demo。

## 数据集

`data/synthetic_bioassay_icrp_v1` 是由固定 ICRP 模型生成的理论验证数据。输入在 `inputs/train|validation|test/*.json`，隐藏真值只在 `labels.jsonl`，正式 planner 不读取隐藏真值。数据中可能有对数噪声、缺失和检测限删失，用于测试反演稳定性与算子成本。

该数据集目前每个核素对应一个真实注册模型，因此模型选择不是主要学习目标；主要学习目标是优化算子和候选点策略。若未来加入同一核素的多个 ICRP route/form 模型，必须通过新的 verified registry 条目实现。

## 当前进度与限制

已完成：固定 ICRP 137 碘/铯模型、来源校验、ICRP v2 检索、MLP gate 接口、确定性 fallback、Qwen operator planner、ODE 数值反演、候选点审计、偏好数据脚本和最小 operator-level GRPO trainer。当前本地 checkpoint 是 20 个训练病例和 5 个验证病例上的 warm-start smoke，不能用于泛化结论。正式 Qwen 主线要求先按训练文档生成 `models/llm_gate_mlp_icrp.pt`；缺少该 checkpoint 时应显式训练，不能回退到旧 gate。

尚未完成：临床验证、剂量学输出、真正 route-specific ingestion/inhalation composite model、把每条 GRPO 候选点在线送入 ODE 重新评分、规模化多 GPU GRPO、专家复核全部自动抽取事实。上述内容不能由 Qwen 自行补齐。

项目的创新点应表述为“外部 ICRP 固定验证器约束下的语言模型数值优化策略学习”，而不是“LLM 发现任意生物动力学模型”。

正式训练命令、目录约定、在线 reward、监控和只读评估见 [`docs/online_training.md`](docs/online_training.md)。
