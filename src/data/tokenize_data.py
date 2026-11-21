"""应用 Qwen chat template，并用 prompt/full token 对齐构建 assistant-only labels。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path('/data/shuaiz/models/Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987')


def encode(row, tokenizer, max_length: int):
    messages = row['messages']
    if not messages or messages[-1].get('role') != 'assistant':
        raise ValueError('each training conversation must end with an assistant message')
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
    full_ids = tokenizer(full_text, add_special_tokens=False)['input_ids']
    # 若模板/tokenizer 不能保证前缀严格对齐，停止处理而非猜测标签边界。
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError('prompt token IDs are not a prefix of full-conversation IDs')
    if len(prompt_ids) >= len(full_ids):
        raise ValueError('assistant response has no target tokens')
    input_ids = full_ids[:max_length]
    labels = [-100] * min(len(prompt_ids), len(input_ids))
    labels += input_ids[len(labels):]
    labels = labels[:len(input_ids)]
    if not any(x != -100 for x in labels):
        raise ValueError('truncation removed every assistant target token')
    return {
        'input_ids': input_ids,
        'attention_mask': [1] * len(input_ids),
        'labels': labels,
        'prompt_length': min(len(prompt_ids), len(input_ids)),
        'full_length': len(full_ids),
        'truncated': len(full_ids) > max_length,
        'assistant_target_tokens': sum(x != -100 for x in labels),
        'rendered_text': full_text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-length', type=int, default=512)
    ap.add_argument('--model', type=Path, default=MODEL)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    out_dir = ROOT / 'data/processed'
    stats = ['# Tokenization 与 assistant-only Loss Mask 检查', '',
             f'- 模型目录：`{args.model}`', f'- 最大序列长度：{args.max_length}',
             '- 标签边界：分别 token 化 assistant generation prompt 与完整对话，并要求 prompt token IDs 是完整 token IDs 的严格前缀。',
             '- prompt（system/user/assistant 起始标记）标签为 `-100`；assistant 回答及结尾 `<|im_end|>` 参与 loss。', '']
    for split in ('train', 'validation', 'test'):
        src = ROOT / f'data/{split}.jsonl'
        rows = [json.loads(line) for line in src.read_text(encoding='utf-8').splitlines() if line.strip()]
        encodings = [encode(row, tok, args.max_length) for row in rows]
        dst = out_dir / f'{split}_tokenized.jsonl'
        with dst.open('w', encoding='utf-8') as f:
            for item in encodings:
                f.write(json.dumps({k:v for k,v in item.items() if k != 'rendered_text'}, ensure_ascii=False) + '\n')
        lengths = [len(x['input_ids']) for x in encodings]
        target_counts = [x['assistant_target_tokens'] for x in encodings]
        truncated = sum(x['truncated'] for x in encodings)
        stats.append(f'## {split}')
        stats.append(f'- 样本数：{len(rows)}；长度 min/mean/max：{min(lengths)}/{mean(lengths):.1f}/{max(lengths)} tokens；目标 token 均值：{mean(target_counts):.1f}；截断：{truncated}')
        stats.append(f'- 输出：`{dst}`')
        stats.append('')
    (ROOT / 'reports/tokenization_report.md').write_text('\n'.join(stats)+'\n', encoding='utf-8')
    print('\n'.join(stats))

if __name__ == '__main__':
    main()
