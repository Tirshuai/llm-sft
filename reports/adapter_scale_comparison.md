# LoRA Adapter 强度对照

- 数据：现有 validation split（30 条）；测试集未读取。
- 对照：Base（scale=0）以及 LoRA scale 0.25/0.5/0.75/1.0。
- 生成：贪心解码，max_new_tokens=96，repetition_penalty=1.1，no_repeat_ngram_size=3。
- 每个问题、每个 scale 使用完全相同的 prompt 和解码参数。
- 质量仍需人工查看逐题输出；验证 NLL 不能单独代表回答质量。

| Adapter scale | Validation NLL |
|---:|---:|
| 0.00 | 3.0256 |
| 0.25 | 2.6459 |
| 0.50 | 2.4589 |
| 0.75 | 2.3815 |
| 1.00 | 2.3591 |

- 逐题输入、参考答案、生成结果和停止原因：`/home/shuaiz/llm-sft-project/reports/adapter_scale_comparison.jsonl`
- 对照耗时：152.6 秒。

## 实际生成观察

30 条验证题中，没有生成在 `<|im_end|>` 处停止；按 scale 0/0.25/0.50/0.75/1.00，分别有 9/6/9/11/13 条达到 max_new_tokens，其余由 `<|endoftext|>` 停止。对“什么是过拟合？”的对照显示，scale 1 的定义更贴近参考答案，但末尾仍输出 `-wsj` 和无关解释；较低 scale 也有乱码续写。因此，NLL 随 scale 增大而下降，并未转化为稳定、干净的回答。测试集未使用。
