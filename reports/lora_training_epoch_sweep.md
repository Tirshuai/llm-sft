# LoRA 训练报告

- 模式：单卡 BF16 LoRA SFT；优化器只接收 LoRA 参数。
- 基础模型：`/data/shuaiz/models/Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987`
- 训练/验证样本：100/30
- 参数：epochs=6，batch=2，grad_accum=4，max_length=512
- 优化：AdamW，lr=0.0002，weight_decay=0.01，warmup_ratio=0.1，scheduler=cosine
- LoRA：r=16，alpha=32，dropout=0.05，targets=['q_proj', 'k_proj', 'v_proj', 'o_proj']
- 可训练参数：2,162,688 / 496,195,456
- 总耗时：38.4 秒；峰值显存：1.54 GiB

| Epoch | Train loss | Validation loss | Learning rate | Global step |
|---:|---:|---:|---:|---:|
| 1 | 3.1035 | 2.5131 | 0.000196 | 13 |
| 2 | 2.5972 | 2.3849 | 0.000167 | 26 |
| 3 | 2.2878 | 2.3535 | 0.000115 | 39 |
| 4 | 2.0457 | 2.3515 | 5.92e-05 | 52 |
| 5 | 1.8885 | 2.3813 | 1.61e-05 | 65 |
| 6 | 1.8238 | 2.3878 | 0 | 78 |

Loss labels 使用 `-100` 屏蔽非 assistant token；causal-LM 的标签移位由模型内部完成。
