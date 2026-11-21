# Base / LoRA / QLoRA Test 对比

- 保留 test split 共 30 条；只输入 user 消息，参考答案不进入 prompt。
- 统一 greedy、max_new_tokens=96、repetition_penalty=1.1、no_repeat_ngram_size=3；停止 ID 包括 EOS 与 `<|im_end|>`。
- LoRA 为 BF16 Base + adapter；QLoRA 为 4-bit NF4 Base + adapter。

| Model | samples | im_end stops | eos stops | max token stops |
|---|---:|---:|---:|---:|
| Base | 30 | 0 | 17 | 13 |
| LoRA | 30 | 0 | 14 | 16 |
| QLoRA | 30 | 0 | 19 | 11 |

逐条输出见 `reports/generation_comparison.jsonl`。停止原因不是回答质量分数。

## 定性抽查

- 步道题答案为 1.3 公里：Base 输出 2.3；LoRA 输出 0.3 并续写另一题；QLoRA 算式和单位混乱。
- 调查摘要题：Base 偏成长篇建议；LoRA 提到范围限制但重复扩写；QLoRA 主体较短但追加无关文本。
- 仅抽查两条，不足以排名；三种模型当前都有明显生成问题，且都没有以 `<|im_end|>` 结束。
