# 后训练调优实验日志

本日志记录当前实验的环境、数据、训练、推理现象和后续对照。具体完整参数见相应配置与报告；输出来自实际运行，不根据预期补写结果。

## 环境与模型

- 服务器：Ubuntu 22.04.3；10 张 NVIDIA RTX 4090 D，首轮只用 `cuda:0`。
- Conda 环境：`/data/shuaiz/conda_envs/llm_sft`，Python 3.11.18。
- 主要库：PyTorch 2.8.0+cu128、Transformers 4.57.1、PEFT 0.20.0、TRL 0.25.0、Accelerate 1.10.1、bitsandbytes 0.45.5。
- 基础模型：Qwen/Qwen2.5-0.5B Base，HF revision `060db6499f32faf8b98477b0a26969ef7d8b9987`；本地权重 SHA-256 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`。权重从 HF 镜像开始下载后停滞，后通过 ModelScope 对相同 SHA-256 的文件断点续传完成。
- 原始 Base 权重未被训练过程修改。LoRA adapter 单独保存在 `outputs/lora/experiment_001`。

## 数据与管线

- 自建中文样本 160 条，10 个主题；固定 seed 2026 划分为 train/validation/test = 100/30/30。
- 数据处理报告：`reports/data_processing_report.md`；tokenization 报告：`reports/tokenization_report.md`。
- Assistant-only mask 通过逐样本 prompt/full token 前缀校验；system/user 及 assistant 头部为 -100，assistant 回答和 `<|im_end|>` 参与 loss。160 条完整 token 序列全部通过 mask 断言；一次带 labels 的 causal-LM 前向 loss 有限。
- 当前 validation split 已用于每 epoch 验证 loss，也用于过拟合主题的定性观察；后续 scale 选择应以 validation 为主，最终测试集保留到决定方案后再用。

## LoRA 实验 001

- 参数：3 epochs；batch 2；gradient accumulation 4；LR 2e-4；AdamW；cosine scheduler；BF16；LoRA r=16、alpha=32、dropout=0.05，targets q/k/v/o。
- 可训练参数 2,162,688 / 496,195,456（0.436%）；全程单卡；峰值显存 1.54 GiB；耗时约 19.7 秒。
- Train loss：3.0458 → 2.5495 → 2.3795。
- Validation loss：2.4858 → 2.3904 → 2.3591。
- Adapter 可重新加载；基础权重 SHA-256 与训练前一致。
- 定性观察：Base 与 LoRA 对算术、过拟合等提示均出现过续写杂乱内容。LoRA 对过拟合问题的主体回答相关，但出现 `-wsj` 等普通文本片段；对“1加5”曾回答错误。当前只可判断训练流程成功、回答质量尚无可靠提升；loss 下降不能当作能力提升证据。
- 原因尚未确定。观察到一些生成未输出 `<|im_end|>`，而继续到其他文本或 token 上限；这与内容本身是否正确需分开评估。

## Adapter 强度对照（已完成）

- 使用 validation 30 条；Base + adapter scale 0、0.25、0.5、0.75、1.0；固定贪心解码、max_new_tokens=96、repetition_penalty=1.1、no_repeat_ngram_size=3。每题/每档提示一致。
- Validation token-weighted NLL：scale 0=3.0256，0.25=2.6459，0.5=2.4589，0.75=2.3815，1.0=2.3591。NLL 随 adapter scale 增加而下降；这说明对参考答案 token 的拟合增强，不能单独说明生成质量提高。
- 30 条生成中所有 scale 均有输出未结束于 `<|im_end|>`：scale 0/0.25/0.50/0.75/1.00 分别有 9/6/9/11/13 条达到 max_new_tokens；其余以 `<|endoftext|>` 结束。没有一条在 `<|im_end|>` 处停止。
- 对“什么是过拟合？”同题对照：scale 0 的内容相对可读但继续生成乱码/追问；scale 1 的定义更贴近 validation 参考答案，但追加 `-wsj` 和错误的过采样/欠采样解释。中间 scale 也出现不自然 token 和续写。因此本次强度扫描没有找到可靠、干净的生成强度；NLL 最佳的 scale=1 不是回答质量最佳的证据。
- adapter 文件及 Base 权重均未修改；测试集未读取。逐题结果在 `reports/adapter_scale_comparison.jsonl`，汇总在 `reports/adapter_scale_comparison.md`。

## `<|im_end|>` 诊断（已完成）

- 检查发现 train/validation/test 分别 100/100、30/30、30/30 样本在 assistant-only labels 中包含 `<|im_end|>`（ID 151645），标签尾部结构正确；数据 mask 未漏掉结束标记。
- tokenizer eos `<|endoftext|>` 是 ID 151643，Base 默认 generation_config 只列 ID 151643；推理脚本已显式把 151643 和 `<|im_end|>` 151645 都作为停止 ID。
- 在 30 条 validation 的参考答案结尾上下文上，Base 的平均 P(`<|im_end|>`) 约 3.22e-6、中位排名 43,084.5；LoRA 的概率约 8.47e-5、中位排名 1,954。LoRA 已朝结束标记方向移动，但结束 token 仍非高概率。相同位置 Base 的 P(`<|endoftext|>`) 约 0.0987，而 LoRA 降至 0.000446。
- scale 对照的 150 条生成中没有一条由 `<|im_end|>` 停止；发生了 EOS 停止或达到 token 上限。reference-context 统计不能完全代表自由生成时的模型状态。
- Base 权重 SHA-256 仍为 `88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`。LoRA 训练参数 2,162,688 / 496,195,456 = 0.436%，但小比例 adapter 仍可显著改变关键 logits；结束 token 概率变化已实测。
- 当前结论：未发现数据/label 遗漏；较可能是 Base 的结束 token 先验、样本少和自由生成分布偏移共同导致停止不稳定，尚不能证明唯一原因。详细统计见 `reports/end_marker_diagnostic.md`。

## 推理脚本强度参数（已完成）

- `src/inference/inference.py` 新增 `--adapter-scale`，允许 0 到 1；使用 PEFT `rescale_adapter_scale` 临时调整 adapter scaling，不改写 adapter 或 Base。语法检查通过。
- 同一提示“为我解释一下过拟合。”以 `max_new_tokens=64` 实测 scale 0/0.5/1：
  - 0：Base 主体回答连贯，随后续写 `Comey` 等内容。
  - 0.5：定义基本连贯，但出现“过拟合力证”等异常文本，随后输出 `<quote>`。
  - 1：回答贴近 SFT 风格，随后重复 `-wsj`。
- 这次演示确认参数会改变生成；没有一个 scale 完全消除续写问题。该比较是定性示例，不替代完整测试集评估。

## Epoch 1/3/6 对照（已完成）

- 为控制同一优化轨迹，使用固定 seed 2026 从新初始化运行一次 6-epoch LoRA，在每个 epoch 保存 adapter checkpoint；从 epoch 1/3/6 快照比较。现有 experiment_001（3 epoch）未覆盖。
- 新轨迹沿用 LR=2e-4、AdamW + cosine、warmup_ratio=0.1、batch=2、grad_accum=4、r=16、target q/k/v/o。因总计划步数为 78，cosine schedule 与原 experiment_001 的 39 步 schedule 不同；本次 1/3/6 快照在同一轨迹内可比，但不要把新 epoch-3 快照当作旧 experiment_001 的精确复现。
- 训练 loss 持续下降：epoch1 3.1035，epoch3 2.2878，epoch6 1.8238。
- Validation loss：epoch1 2.5131，epoch3 2.3535，epoch4 最低 2.3515，epoch5 2.3813，epoch6 2.3878。第 4 轮后训练继续下降而验证上升，出现验证分布上的过拟合迹象。
- 固定生成对照：validation 题“什么是过拟合？”和一条外部探针“后训练是什么？”，相同 greedy 解码，比较 epoch 1/3/6。三档都出现乱码/错误概念/无关续写；没有观察到更多 epochs 修复语言能力或停止标记。此为少量定性问题，不是完整能力评测。
- 6 个演示生成都没有由 `<|im_end|>` 停止；其中一些到 max_new_tokens，其余到 `<|endoftext|>`。
- 结果：更多训练轮次使训练拟合继续增强；validation 在第 4 轮后恶化；当前生成质量没有可靠改善。因此“加 epoch”不是这轮问题的修复办法。
- Checkpoints：`outputs/lora/epoch_sweep_6epoch/checkpoints/epoch_01` 到 `epoch_06`；最终 epoch-6 adapter 在 `outputs/lora/epoch_sweep_6epoch`。
- 报告：`reports/lora_training_epoch_sweep.md`；固定提示生成：`reports/epoch_generation_comparison.jsonl` 和 `.md`。测试集未读取。

## Epoch sweep runner（脚本已准备）

- `src/training/train.py` 现在支持 `--epochs`、`--output-dir`、`--report-path` 和 `--save-epoch-checkpoints`，语法检查与 `--help` 检查通过。
- `scripts/run_epoch_sweep.sh` 可运行多个独立 epoch 配置，例如 `bash scripts/run_epoch_sweep.sh 1 3 6`。每个 run 写入独立目录和报告；训练脚本遇到非空输出目录会停止，防止覆盖。
- 该 helper 为不同总 epoch 数分别计算 cosine 学习率调度周期；若需严格比较同一轨迹中的 checkpoints，使用现有 `outputs/lora/epoch_sweep_6epoch/checkpoints/epoch_01/03/06`。
- 本节只记录脚本准备和静态检查，本轮未启动新的训练。


## QLoRA 实验（已完成）

- 4-bit NF4、double quantization、BF16 compute；可训练参数 2,162,688。报告总参数 317,282,176 是量化参数计数口径，不等于显存占用。
- 3 epochs、39 updates，18.9 秒，峰值显存 1.45 GiB。Train loss 3.0926 → 2.3823；validation loss 2.5410 → 2.4518。
- Smoke check 确认实际 4-bit 加载、存在 Linear4bit 层且前向 loss 有限；随后完成训练。详见 `reports/qlora_training_report.md`。

## Held-out Base / LoRA / QLoRA 对照（已完成）

- Test split 30 条分别输入 Base BF16、LoRA BF16、QLoRA NF4；reference 未进入 prompt。逐条输出见 `reports/generation_comparison.jsonl`。
- 停止统计（im_end / EOS / 96 token 上限）：Base 0/17/13，LoRA 0/14/16，QLoRA 0/19/11。
- 两题抽查发现算术错误、串题、重复扩写和无关文本。样本太少不能排名；当前无证据证明 adapter 提升整体生成质量。按用户决定，不继续在 0.5B 上调质量，转向整理工程记录。

## 项目文档与 GitHub 准备

- 新增 README、固定核心依赖版本的 requirements.txt 和排除模型/训练产物/缓存/密钥的 .gitignore。
- 项目此前无 Git 元数据。本轮未初始化 Git、配置远端或推送 GitHub。
