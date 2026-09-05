# SkyRescue-Bench v1.0.0

这是 SkyRescue 论文配套的公开数据发布包，面向低空应急任务的智能体工作流编译、实体锚定、运行时保障和双模型复现实验。数据均为合成场景或经匿名化处理的专家复核数据，不代表真实飞行记录，也不构成飞行安全认证。

## 内容

- `data/human_instructions_100/`: 100 条人工编写指令及专家裁决后的 7 字段金标准。
- `data/human_instructions_200/`: 200 条 AI 辅助预标注、双专家复核并裁决后的匿名发布版。
- `data/entity_grounding_heldout100/`: 100 条确证性盲测输入、冻结金标准及标注协议。
- `results/llm_confirmatory/`: DeepSeek/Qwen 的 HeldOut100 原始响应、响应内容和指标文件。
- `results/llm_replication/`: 100 条人工指令上的双模型复现实验结果。
- `metadata/`: 数据说明、标注一致性摘要、模型配置、响应哈希和引用信息。

## 公开边界

公开包不包含专家姓名、专家代码、标注日期、裁决签名、原始审计工作簿、API 密钥或本机绝对路径。A/B 原始标注和完整裁决工作簿由作者单独留存，仅在审稿或复核需要时按约定提供。

## 使用顺序

1. 先阅读 `metadata/DATASHEET.md` 和 `metadata/ANNOTATION_AUDIT_SUMMARY.csv`。
2. 使用 `data/entity_grounding_heldout100/*BlindInput*` 作为模型输入；评分时再读取 Gold Standard。
3. 对模型响应使用 `results/*/predictions.csv` 和 `summary.*`，不要把金标准混入模型提示词。
4. 使用根目录 `SHA256SUMS.txt` 验证文件完整性。

## 许可证

数据集建议按 CC BY 4.0 使用，具体条款见 `LICENSE.txt`。代码不在本数据包中，代码仓库见：<https://github.com/liuyushugreat/SkyNetUamPlatform/tree/main/modules/Skyrescue>。

## 联系与引用

Zenodo DOI 在正式发布后补入论文和本说明。发布前请不要把本地路径或 API 密钥写入公开包。
