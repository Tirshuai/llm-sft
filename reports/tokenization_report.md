# Tokenization 与 assistant-only Loss Mask 检查

- 模型目录：`/data/shuaiz/models/Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987`
- 最大序列长度：512
- 标签边界：分别 token 化 assistant generation prompt 与完整对话，并要求 prompt token IDs 是完整 token IDs 的严格前缀。
- prompt（system/user/assistant 起始标记）标签为 `-100`；assistant 回答及结尾 `<|im_end|>` 参与 loss。

## train
- 样本数：100；长度 min/mean/max：41/62.7/92 tokens；目标 token 均值：26.7；截断：0
- 输出：`/home/shuaiz/llm-sft-project/data/processed/train_tokenized.jsonl`

## validation
- 样本数：30；长度 min/mean/max：51/63.0/81 tokens；目标 token 均值：26.6；截断：0
- 输出：`/home/shuaiz/llm-sft-project/data/processed/validation_tokenized.jsonl`

## test
- 样本数：30；长度 min/mean/max：41/61.4/79 tokens；目标 token 均值：25.8；截断：0
- 输出：`/home/shuaiz/llm-sft-project/data/processed/test_tokenized.jsonl`

