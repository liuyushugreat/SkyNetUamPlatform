# SkyRescue-Bench 数据说明书

## 1. 动机

数据集用于评估安全关键智能体信息系统中的自然语言任务编译、实体锚定、外部动作提交和运行时保障。它服务于软件机制研究，不用于训练或认证真实无人机控制器。

## 2. 数据构成

| 子集 | 规模 | 来源与用途 | 公开形式 |
|---|---:|---|---|
| HumanInstructions-100 | 100 条 | 人工编写、双专家独立标注、第三专家裁决；用于真实指令候选任务生成实验 | CSV/JSONL |
| ManualInstructions-200 | 200 条 | AI 辅助预标注后由两位专家复核并裁决；用于补充分布覆盖 | 匿名 XLSX |
| EntityGrounding-HeldOut100 | 100 条 | 独立人工编写、双专家标注、第三专家裁决；用于冻结确证性盲测 | CSV/JSONL/XLSX |
| LLM results | 2 个模型、各 100 条 | 固定模型、提示词、temperature=0、每条 1 次 | JSONL/CSV/JSON |

## 3. 标注与质量

HumanInstructions-100 的宏平均 Cohen's kappa 为 0.8731，微观精确一致率为 93.86%。EntityGrounding-HeldOut100 的宏平均 Cohen's kappa 为 0.9533，微观精确一致率为 97.43%。200 条数据是 AI 辅助专家复核数据，不应表述为两位专家从零独立创作。

## 4. 标签

每条指令包含 `task_type`、`target_zone`、`priority`、`deadline_s_or_text`、`required_skill`、`needs_human_approval` 和 `expected_failure` 七个结构化字段。地点标签采用严格文本匹配或冻结实体锚定规则，不能把 Schema 通过解释为语义正确。

## 5. 已知限制

场景为合成或人工编写文本，不能代表真实飞行分布；地点实体存在歧义；专家标注包含 AI 辅助复核过程；LLM 结果受模型版本、提示词和服务端实现影响；公开数据不能证明真实飞行安全、开放攻击防御或监管合规。

## 6. 隐私与敏感信息

公开发布前已排除专家身份字段、标注日期、裁决签名、API 密钥和本机路径。若未来发现个人信息或敏感地点，应先撤回或发布新版本。

## 7. 可复现性

模型响应以 JSONL 形式保存，实验配置固定为 temperature=0、top_p=1、max_tokens=512、每条指令运行 1 次。响应哈希和提示词哈希见 `metadata/LLM_RESPONSE_HASHES.json`。
