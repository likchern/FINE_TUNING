# 评测套件（题目交付项 4）

本目录为 **基于强化学习的多轮对话策略优化** 项目中 **交付项 4：全套评测套件（自动化脚本 + 人工评审模板）** 的落地实现，用于对生成的多轮对话数据进行质量验证。数据格式与 `RL/data/test_dia.jsonl` 一致。

## 目录结构

```
evaluation/
├── README.md                    # 本说明
├── validate_dialogue_data.py    # 自动化校验脚本
├── app_streamlit.py            # 校验结果前端（Streamlit）
├── human_review_template.csv   # 人工评审表（Excel/CSV）
├── human_review_guide.md        # 人工评审维度说明与使用指南
└── requirements.txt             # 依赖（含 streamlit）
```

## 1. 自动化脚本

### 校验范围（不止格式与基础质量）

自动化脚本对以下多类项做校验，**默认即做**的包括：

| 类别 | 内容 |
|------|------|
| **格式与结构** | 每行合法 JSON；必填顶层字段（`dialogue_id`、`patient_id`、`persona`、`topic`、`dialogue_history`、`metadata`）；`dialogue_history` 为列表且每条含 `role`、`content`；首轮为用户发言；`persona` 含 `raw_persona` 或 `enhanced_description`。 |
| **内容质量** | 单轮 `content` 不得为空；不得含占位符或未替换模板（如 `[内容]`、`Response:`/`Thinking:` 泄露、`{{...}}` 等）。 |
| **对话流程** | 对话轮次不少于 2 轮；`metadata.total_turns` 与 `dialogue_history` 长度一致。 |
| **统计与提示** | 对话轮次分布、安全相关关键词命中次数、连续三轮同角色等流程提示（仅统计/提示，不直接判不通过）。 |

**十维度评分**（与 `human_review_guide.md` 一致）：

- **规则代理**：`--score-dimensions` 时，对每条通过校验的对话按十个维度打 0/1/2 分（规则与启发式），并输出各维度均值及总分均值。
- **Qwen API（LLM）**：`--score-llm` 时，改为调用 Qwen 模型对每条对话进行十维度+总分评分。需配置 API Key：
  - 环境变量 **`QWEN_API_KEY`** 或 **`DASHSCOPE_API_KEY`**；
  - 或在项目根目录（RL/）下创建 **`.env`** 文件，写 `QWEN_API_KEY=你的key`（若已安装 `python-dotenv` 会自动加载）。
- 模型：默认 **`qwen-plus`**，可用 `--model qwen-turbo` 等覆盖。
- 使用 `--output-scores <csv>` 可导出每条对话的十维度+总分 CSV。

**可选严格模式**（加参数后才判不通过）：

- **长度**（`--strict-length`）：用户单轮 ≤40 字、照护师单轮 ≤80 字，超限则判不通过。
- **安全**（`--strict-safety`）：助手回复中出现风险表述关键词（如“绝对治愈”“不必用药”等）时判不通过。

### 使用

```bash
# 在项目根目录 RL/ 下执行，或指定路径
cd /home/zy/RL
python evaluation/validate_dialogue_data.py data/test_dia.jsonl

# 将报告写入 JSON
python evaluation/validate_dialogue_data.py data/test_dia.jsonl -o evaluation/report.json

# 仅输出通过/未通过与计数（便于 CI）
python evaluation/validate_dialogue_data.py data/test_dia.jsonl --quiet

# 严格长度校验：用户单轮≤40 字、助手单轮≤80 字，超限则判不通过
python evaluation/validate_dialogue_data.py data/test_dia.jsonl --strict-length

# 严格安全校验：助手回复含风险表述关键词则判不通过
python evaluation/validate_dialogue_data.py data/test_dia.jsonl --strict-safety

# 十维度自动化评分（规则）+ 导出评分 CSV
python evaluation/validate_dialogue_data.py data/test_dia.jsonl --score-dimensions --output-scores evaluation/auto_scores.csv

# 十维度评分改为调用 Qwen API（需配置 QWEN_API_KEY 或 .env）
python evaluation/validate_dialogue_data.py data/test_dia.jsonl --score-dimensions --score-llm --model qwen-plus
# 直接运行这个就行，同时得到：通过/不通过判断 + 通过条目的 LLM 得分 CSV；加 -o 可保存完整报告供前端展示
python evaluation/validate_dialogue_data.py data/new_dialogues_600.jsonl --score-dimensions --score-llm --model qwen-turbo -o evaluation/report.json --output-scores evaluation/auto_scores_llm.csv
```

- 退出码：全部通过为 0，存在未通过为 1，文件不存在等为 2。
- 默认已做格式、结构、内容质量（空内容、占位符）、流程与 metadata 一致性的校验；长度与安全可按需加 `--strict-length`、`--strict-safety`。

### 校验结果前端（Streamlit）

在项目根目录执行：

```bash
pip install -r evaluation/requirements.txt   # 含 streamlit
python3 -m streamlit run evaluation/app_streamlit.py
```

若安装时提示 `streamlit` 在 `~/.local/bin` 且未加入 PATH，可直接用 `python3 -m streamlit` 避免找不到命令。

- **加载已有报告**：左侧选择「加载已有校验报告」，填写 report.json 路径（默认 `evaluation/report.json`），可查看每条通过/未通过及分数或问题。
- **重新校验**：选择「重新校验」，填写 jsonl 路径后点击「开始校验」，使用规则评分（不调 API）生成结果并展示。
- 每条数据：通过则展示十维度+总分；未通过则展示具体未通过原因。支持按「全部 / 仅通过 / 仅未通过」筛选。

## 2. 人工评审模板

- **human_review_template.csv**：表头为 `dialogue_id` 与 10 个评分维度（即时性、意图识别、内容准确性、建议可遵循、恰当表达、情感关怀、话题引导、安全性、负向情感风险、合规性），每维度 0/1/2 分，最后一列为备注。
- **human_review_guide.md**：说明各维度定义、2/1/0 分标准、使用步骤，以及与自动化脚本的配合方式。

使用流程建议：

1. 用自动化脚本对目标 `.jsonl` 做格式与基础质量校验。
2. 对通过的数据按需抽样，在模板中填写 `dialogue_id` 与各维度分数。
3. 汇总人工评分用于质量报告或奖励模型训练。

## 3. 数据格式约定

评测对象为与 `data/test_dia.jsonl` 同构的 jsonl：

- 每行一条 JSON。
- 每条至少包含：`dialogue_id`、`patient_id`、`persona`（含 `raw_persona` 或 `enhanced_description`）、`topic`、`dialogue_history`（列表）、`metadata`。
- `dialogue_history` 中每项为 `{"role": "user"|"assistant", "content": "..."}`，可含 `thinking`（助手）。

## 4. 与题目奖励函数的关系

人工评审的 10 个维度与题目中 **3.2 奖励函数评分维度** 一致，可用于：

- 对单条或整轮对话的助手回复做质量标注；
- 与自动化脚本的格式/长度/安全提示一起，构成完整的数据质量评测套件。
