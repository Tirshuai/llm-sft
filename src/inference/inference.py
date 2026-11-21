"""加载本地 Qwen Base + LoRA Adapter 推理；支持单条问题或交互模式。"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from peft import PeftModel
from peft.helpers import rescale_adapter_scale
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = Path('/data/shuaiz/models/Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987')
DEFAULT_ADAPTER = ROOT / 'outputs/lora/experiment_001'


def load_model(base_path: Path, adapter_path: Path):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; this script is configured for GPU inference')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This run expects the BF16 support verified on the training GPU')
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    ).to('cuda:0')
    # adapter 是增量权重；加载时叠加在原始 Base 上，不会改写 Base 文件。
    model = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def answer(messages, tokenizer, model, args):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors='pt',
        return_dict=True,
    ).to('cuda:0')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    stop_ids = [tokenizer.eos_token_id]
    if im_end_id not in stop_ids:
        stop_ids.append(im_end_id)
    generation_args = {
        'max_new_tokens': args.max_new_tokens,
        'do_sample': args.sample,
        'eos_token_id': stop_ids,
        'pad_token_id': tokenizer.pad_token_id or tokenizer.eos_token_id,
        'repetition_penalty': args.repetition_penalty,
        'no_repeat_ngram_size': args.no_repeat_ngram_size,
    }
    if args.sample:
        generation_args.update(temperature=args.temperature, top_p=args.top_p)
    with rescale_adapter_scale(model, args.adapter_scale):
        with torch.inference_mode():
            output_ids = model.generate(**inputs, **generation_args)
    new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', type=Path, default=DEFAULT_BASE)
    parser.add_argument('--adapter', type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument('--adapter-scale', type=float, default=1.0, help='LoRA strength multiplier in [0, 1]; 0 approximates Base, 1 uses the saved adapter at full scale')
    parser.add_argument('--prompt', type=str, help='单条提问；省略后进入交互模式')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--repetition-penalty', type=float, default=1.1)
    parser.add_argument('--no-repeat-ngram-size', type=int, default=3)
    parser.add_argument('--sample', action='store_true', help='启用采样；默认使用贪心解码')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    args = parser.parse_args()
    if not 0.0 <= args.adapter_scale <= 1.0:
        parser.error('--adapter-scale must be between 0 and 1')
    if args.max_new_tokens < 1:
        parser.error('--max-new-tokens must be >= 1')
    if args.sample and args.temperature <= 0:
        parser.error('--temperature must be > 0 when --sample is set')
    tokenizer, model = load_model(args.base_model, args.adapter)
    print(f'Base: {args.base_model}')
    print(f'Adapter: {args.adapter} (scale={args.adapter_scale:g})')

    if args.prompt is not None:
        messages = [{'role': 'user', 'content': args.prompt}]
        print(answer(messages, tokenizer, model, args))
        return

    history = []
    print('交互模式：输入 :reset 清空上下文，输入 :quit 退出。')
    while True:
        try:
            prompt = input('\n你：').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n退出。')
            break
        if not prompt:
            continue
        if prompt == ':quit':
            break
        if prompt == ':reset':
            history.clear()
            print('上下文已清空。')
            continue
        history.append({'role': 'user', 'content': prompt})
        response = answer(history, tokenizer, model, args)
        print(f'模型：{response}')
        history.append({'role': 'assistant', 'content': response})


if __name__ == '__main__':
    main()
