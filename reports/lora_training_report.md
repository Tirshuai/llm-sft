# LoRA 训练报告

- 模式：单卡 BF16 LoRA SFT；优化器只接收 LoRA 参数。
- 基础模型：`/data/shuaiz/models/Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987`
- 训练/验证样本：100/30
- 参数：epochs=3，batch=2，grad_accum=4，max_length=512
- 优化：AdamW，lr=0.0002，weight_decay=0.01，warmup_ratio=0.1，scheduler=cosine
- LoRA：r=16，alpha=32，dropout=0.05，targets=['q_proj', 'k_proj', 'v_proj', 'o_proj']
- 可训练参数：2,162,688 / 496,195,456
- 总耗时：19.7 秒；峰值显存：1.54 GiB

| Epoch | Train loss | Validation loss | Learning rate | Global step |
|---:|---:|---:|---:|---:|
| 1 | 3.0458 | 2.4858 | 0.000164 | 13 |
| 2 | 2.5495 | 2.3904 | 5.77e-05 | 26 |
| 3 | 2.3795 | 2.3591 | 0 | 39 |

Loss labels 使用 `-100` 屏蔽非 assistant token；causal-LM 的标签移位由模型内部完成。
