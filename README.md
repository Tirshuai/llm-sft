# Qwen2.5-0.5B 中文 SFT / LoRA / QLoRA 实验

项目覆盖中文指令数据处理、assistant-only loss mask、LoRA/QLoRA 训练、adapter 推理及 held-out 对照。重点是学习和记录流程，不是产出高质量助手。

## 本轮结论

- 160 条样本按 seed 2026 划分为 train/validation/test = 100/30/30。
- LoRA、QLoRA 的训练 loss 均下降，但不代表生成能力提升；6 epoch LoRA 轨迹在 epoch 4 达到最低 validation loss，随后回升。
- Base、LoRA、QLoRA 各生成 30 条测试题。都未以 `<|im_end|>` 停止；抽查见到算术错误、乱码、串题及无关扩写。没有证据证明 adapter 提升了整体质量。详见 `reports/experiment_log.md` 和 `reports/evaluation_report.md`。

## 目录

- `configs/`：LoRA、QLoRA 和 epoch sweep 配置。
- `data/raw/`：种子数据；`data/{train,validation,test}.jsonl`：划分数据。
- `src/data/`：数据生成、校验、tokenize 和 mask 可视化。
- `src/training/train.py`：BF16 LoRA / 4-bit NF4 QLoRA。
- `src/inference/inference.py`：单题和交互推理。
- `src/evaluation/`：adapter scale、epoch、held-out 对照。
- `reports/`：完整实验报告；`outputs/`：adapter/checkpoint（Git 忽略）。

## 环境和模型

本次已验证 Ubuntu 22.04、RTX 4090 D、Python 3.11.18、PyTorch 2.8.0 CUDA 12.8 wheel。版本见 `requirements.txt` 与环境报告。

```bash
conda create -n llm_sft python=3.11 -y
conda activate llm_sft
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
python -m pip install -r requirements.txt
```

Base 不随仓库分发。本次为固定 revision 的 Qwen2.5-0.5B；

## 数据准备

从仓库根目录：

```bash
python src/data/build_seed_data.py
python src/data/prepare_data.py --seed 2026
python src/data/tokenize_data.py --max-length 512
```

管线检查 JSONL、去重、划分和模板前缀对齐。prompt 标签为 -100，assistant 回答及 `<|im_end|>` 参与 loss。

## 训练

```bash
python src/training/train.py --config configs/lora.yaml
python src/training/train.py --config configs/qlora.yaml
python src/training/train.py --config configs/lora.yaml --epochs 6 --output-dir outputs/lora/my_run --report-path reports/my_run.md --save-epoch-checkpoints
bash scripts/run_epoch_sweep.sh 1 3 6
```

默认输出是 `outputs/lora/experiment_001`、`outputs/qlora/experiment_001`。训练拒绝覆盖非空目录。epoch sweep 使用服务器 Conda 路径，迁移后需修改脚本。

## 推理和评估

```bash
python src/inference/inference.py --adapter outputs/lora/experiment_001
python src/inference/inference.py --adapter outputs/lora/experiment_001 --prompt "什么是过拟合？"
python src/inference/inference.py --adapter outputs/lora/experiment_001 --adapter-scale 0.5 --prompt "什么是过拟合？"
python src/evaluation/evaluate.py
```

交互模式支持 `:reset`、`:quit`；scale 在 0 到 1 之间。评估只输入测试问题，不把答案拼入 prompt；greedy、最多 96 个新 token。逐条结果见 `reports/generation_comparison.jsonl`。30 条测试题不足以支持广泛结论。
