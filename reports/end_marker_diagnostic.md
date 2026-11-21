# `<|im_end|>` 数据与生成诊断

## 数据与 labels

检查 tokenized JSONL 中的 `<|im_end|>` token ID（151645）：

| Split | 样本数 | 监督 labels 中含 assistant `<|im_end|>` 的样本 | 命中数 |
|---|---:|---:|---:|
| train | 100 | 100 | 100 |
| validation | 30 | 30 | 30 |
| test | 30 | 30 | 30 |

每条样本末尾 labels 的结构为 assistant 回答最后 token、`<|im_end|>`、换行 token。非 assistant prompt 区域仍为 -100。`<|im_end|>` 不等于 tokenizer 的 `eos_token`：前者 ID 151645，后者 `<|endoftext|>` ID 151643；模型默认 generation_config 的 eos_token_id 是 151643。推理脚本显式把两者都设为停止 ID。

## 参考答案结尾处的下一 token 分布

在 validation 的 reference/full sequence 中，取 assistant `<|im_end|>` 前一个位置的 logits；这等价于检查模型在正确答案之后预测结束 token 的能力。统计 30 条样本：

| Model | P(`<|im_end|>`) 平均值 | `<|im_end|>` 中位排名 | P(`<|endoftext|>`) 平均值 | EOS 中位排名 |
|---|---:|---:|---:|---:|
| Base（scale 0） | 0.00000322 | 43,084.5 | 0.098707 | 6 |
| LoRA（scale 1） | 0.0000847 | 1,954 | 0.000446 | 58,457.5 |

LoRA 在 reference answer 结尾上下文中把概率向 `<|im_end|>` 移动约 26 倍，并显著降低 Base 最偏好的 `<|endoftext|>`。然而 `<|im_end|>` 仍未成为高排名候选。强度扫描的 150 条生成没有任何一条在 `<|im_end|>` 结束；各条要么输出 `<|endoftext|>`，要么达到长度上限。

## 初步判断与限制

- 数据构造、模板对齐和 assistant-only mask 中没有发现遗漏 assistant 结束标记；训练 labels 确实包含该标记。
- 这不证明模型一定能在自由生成时停对。上面的概率是在参考答案结尾上下文上计算；生成时模型走的是自己的文本，略有偏离就会改变后续 logits。
- 一个可能因素是 Base 的预训练更偏向 `<|endoftext|>`，而本项目只有 100 个 assistant 结尾样本。LoRA 已明显改变两个结束 token 的相对偏好，但仍没把 `<|im_end|>` 排到可稳定生成的位置。该解释与测量相符，但不是唯一因果证明。
- Base checkpoint SHA-256 仍为 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`，训练没有改写基础权重。
- LoRA 有 2,162,688 个可训练参数（496,195,456 总参数的 0.436%）；参数比例小不意味着对特定 logits 的影响小。该 adapter 对结束 token 概率的改变就是实测例子。
