"""单卡 BF16 LoRA SFT；labels 中 -100 的位置不参与 causal-LM loss。"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]


class TokenizedJsonl(Dataset):
    """读取 tokenize_data.py 生成的 input_ids / labels。"""
    def __init__(self, path: Path, max_length: int):
        self.path = path
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"dataset is empty: {path}")
        for i, row in enumerate(self.rows):
            n = len(row["input_ids"])
            if n > max_length:
                raise ValueError(f"row {i} exceeds configured max_length={max_length}: {path}")
            if n != len(row["labels"]) or n != len(row["attention_mask"]):
                raise ValueError(f"sequence fields have different lengths: row {i} in {path}")
            if not any(label != -100 for label in row["labels"]):
                raise ValueError(f"row {i} has no supervised tokens in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def make_collator(pad_token_id: int):
    """把 batch 右侧补齐；padding 的 attention=0 且 label=-100。"""
    def collate(rows):
        width = max(len(row["input_ids"]) for row in rows)
        batch_size = len(rows)
        input_ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
        labels = torch.full((batch_size, width), -100, dtype=torch.long)
        for i, row in enumerate(rows):
            length = len(row["input_ids"])
            input_ids[i, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
            attention_mask[i, :length] = torch.tensor(row["attention_mask"], dtype=torch.long)
            labels[i, :length] = torch.tensor(row["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def token_weighted_loss(model, loader, device):
    """按有效预测 token 数加权，避免短 batch 与长 batch 等权。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            target_count = int((batch["labels"][:, 1:] != -100).sum().item())
            loss = model(**batch).loss
            total_loss += float(loss) * target_count
            total_tokens += target_count
    model.train()
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lora.yaml")
    parser.add_argument("--epochs", type=int, help="override num_train_epochs")
    parser.add_argument("--output-dir", type=str, help="override output_dir; use a new directory for each run")
    parser.add_argument("--report-path", type=str, help="override report_path")
    parser.add_argument("--save-epoch-checkpoints", action="store_true", help="save an adapter snapshot after every epoch")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        if args.epochs < 1: parser.error("--epochs must be >= 1")
        cfg["num_train_epochs"] = args.epochs
    if args.output_dir is not None: cfg["output_dir"] = args.output_dir
    if args.report_path is not None: cfg["report_path"] = args.report_path
    if args.save_epoch_checkpoints: cfg["save_epoch_checkpoints"] = True
    resolve = lambda value: (ROOT / value).resolve() if not Path(value).is_absolute() else Path(value)
    model_path = resolve(cfg["model_path"])
    train_path = resolve(cfg["train_file"])
    validation_path = resolve(cfg["validation_file"])
    output_dir = resolve(cfg["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently train on CPU")
    if cfg.get("bf16", True) and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but this GPU/runtime does not support it")

    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda:0")  # 首轮明确单卡，不使用 DDP/多卡自动映射。
    max_length = int(cfg["max_length"])
    train_ds = TokenizedJsonl(train_path, max_length)
    validation_ds = TokenizedJsonl(validation_path, max_length)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    collator = make_collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["per_device_train_batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=int(cfg["per_device_train_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float32
    quant_cfg = cfg.get("quantization", {})
    load_in_4bit = bool(quant_cfg.get("load_in_4bit", False))
    if load_in_4bit:
        compute_name = quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
        compute_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[compute_name]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, quantization_config=bnb_config,
            device_map={"": 0}, local_files_only=True,
        )
        if not getattr(model, "is_loaded_in_4bit", False):
            raise RuntimeError("Requested QLoRA but the base model was not loaded in 4-bit")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        four_bit_layers = sum(module.__class__.__name__ == "Linear4bit" for module in model.modules())
        if four_bit_layers == 0:
            raise RuntimeError("No bitsandbytes Linear4bit modules found after loading")
        print(f"quantization=4bit_{quant_cfg.get('bnb_4bit_quant_type', 'nf4')} linear4bit_layers={four_bit_layers}", flush=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=dtype, local_files_only=True
        ).to(device)
    model.config.use_cache = False
    lora_cfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    ))
    trainable_named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable_named or any("lora_" not in name for name, _ in trainable_named):
        raise RuntimeError("Only LoRA adapter parameters should be trainable")
    trainable_params = [param for _, param in trainable_named]
    trainable_count = sum(param.numel() for param in trainable_params)
    total_count = sum(param.numel() for param in model.parameters())
    print(f"parameters: trainable={trainable_count:,} total={total_count:,} ({100*trainable_count/total_count:.3f}%)", flush=True)

    lr = float(cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        trainable_params, lr=lr, weight_decay=float(cfg["weight_decay"])
    )
    grad_accum = int(cfg["gradient_accumulation_steps"])
    epochs = int(cfg["num_train_epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_updates = updates_per_epoch * epochs
    warmup_steps = max(1, int(total_updates * float(cfg["warmup_ratio"])))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"
    history = []
    global_step = 0
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        train_loss_sum = 0.0
        train_token_count = 0
        for batch_index, cpu_batch in enumerate(train_loader):
            batch = {key: value.to(device, non_blocking=True) for key, value in cpu_batch.items()}
            target_count = int((batch["labels"][:, 1:] != -100).sum().item())
            loss = model(**batch).loss
            train_loss_sum += float(loss.detach()) * target_count
            train_token_count += target_count
            accumulation_start = (batch_index // grad_accum) * grad_accum
            accumulation_size = min(grad_accum, len(train_loader) - accumulation_start)
            (loss / accumulation_size).backward()
            is_update = (batch_index + 1) % grad_accum == 0 or batch_index + 1 == len(train_loader)
            if is_update:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        train_loss = train_loss_sum / max(train_token_count, 1)
        validation_loss = token_weighted_loss(model, validation_loader, device)
        metrics = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "gpu_peak_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "elapsed_seconds": time.time() - start_time,
        }
        history.append(metrics)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        if cfg.get("save_epoch_checkpoints", False):
            checkpoint_dir = output_dir / "checkpoints" / f"epoch_{epoch + 1:02d}"
            model.save_pretrained(checkpoint_dir, safe_serialization=True)
            print(f"epoch_checkpoint={checkpoint_dir}", flush=True)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    elapsed = time.time() - start_time
    peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
    quant_description = (
        f"4-bit {quant_cfg.get('bnb_4bit_quant_type', 'nf4')}, double_quant={quant_cfg.get('bnb_4bit_use_double_quant', True)}, compute={quant_cfg.get('bnb_4bit_compute_dtype', 'bfloat16')}"
        if load_in_4bit else "none (BF16 LoRA)"
    )
    report = [
        "# LoRA/QLoRA 训练报告", "",
        f"- 模式：单卡 {'QLoRA' if load_in_4bit else 'BF16 LoRA'} SFT；优化器只接收 LoRA 参数。",
        f"- Quantization：{quant_description}",
        f"- 基础模型：`{model_path}`",
        f"- 训练/验证样本：{len(train_ds)}/{len(validation_ds)}",
        f"- 参数：epochs={epochs}，batch={cfg['per_device_train_batch_size']}，grad_accum={grad_accum}，max_length={max_length}",
        f"- 优化：AdamW，lr={lr:g}，weight_decay={cfg['weight_decay']}，warmup_ratio={cfg['warmup_ratio']}，scheduler=cosine",
        f"- LoRA：r={lora_cfg['r']}，alpha={lora_cfg['lora_alpha']}，dropout={lora_cfg['lora_dropout']}，targets={lora_cfg['target_modules']}",
        f"- 可训练参数：{trainable_count:,} / {total_count:,}",
        f"- 总耗时：{elapsed:.1f} 秒；峰值显存：{peak_gib:.2f} GiB", "",
        "| Epoch | Train loss | Validation loss | Learning rate | Global step |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in history:
        report.append(
            f"| {item['epoch']} | {item['train_loss']:.4f} | "
            f"{item['validation_loss']:.4f} | {item['learning_rate']:.3g} | {item['global_step']} |"
        )
    report += ["", "Loss labels 使用 `-100` 屏蔽非 assistant token；causal-LM 的标签移位由模型内部完成。", ""]
    report_path = resolve(cfg.get("report_path", "reports/lora_training_report.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"adapter_saved={output_dir}", flush=True)
    print(f"report_saved={report_path}", flush=True)


if __name__ == "__main__":
    main()
