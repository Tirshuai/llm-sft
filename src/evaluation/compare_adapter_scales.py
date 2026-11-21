"""在验证集上比较 Base 与不同 LoRA adapter 强度；不触碰测试集。"""
from __future__ import annotations
import json, time
from collections import Counter
from pathlib import Path
import torch
from peft import PeftModel
from peft.helpers import rescale_adapter_scale
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data/shuaiz/models/Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987')
ADAPTER=ROOT/'outputs/lora/experiment_001'
SCALES=[0.0,0.25,0.5,0.75,1.0]
MAX_NEW_TOKENS=96
BATCH_SIZE=2


def read_rows():
    return [json.loads(x) for x in (ROOT/'data/validation.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]


def encode_prompt(row,tokenizer):
    messages=row['messages'][:-1]
    return tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors='pt',return_dict=True)


def collate(rows,pad_id):
    width=max(len(x['input_ids']) for x in rows)
    ids=torch.full((len(rows),width),pad_id,dtype=torch.long)
    mask=torch.zeros((len(rows),width),dtype=torch.long)
    labels=torch.full((len(rows),width),-100,dtype=torch.long)
    for i,row in enumerate(rows):
        k=len(row['input_ids']); ids[i,:k]=torch.tensor(row['input_ids']); mask[i,:k]=1; labels[i,:k]=torch.tensor(row['labels'])
    return {'input_ids':ids,'attention_mask':mask,'labels':labels}


def validation_nll(model,rows,tokenizer,scale):
    encoded=[]
    for row in rows:
        item=tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=False,return_dict=True)
        prompt=tokenizer.apply_chat_template(row['messages'][:-1],tokenize=True,add_generation_prompt=True,return_dict=True)['input_ids']
        ids=item['input_ids']
        if ids[:len(prompt)]!=prompt: raise ValueError('prompt/full IDs do not align')
        encoded.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':[-100]*len(prompt)+ids[len(prompt):]})
    total=0.; n_tokens=0
    with rescale_adapter_scale(model,scale),torch.inference_mode():
        for start in range(0,len(encoded),BATCH_SIZE):
            batch=collate(encoded[start:start+BATCH_SIZE],tokenizer.pad_token_id).copy()
            batch={k:v.to('cuda:0') for k,v in batch.items()}
            n=int((batch['labels'][:,1:]!=-100).sum())
            loss=model(**batch).loss
            total+=float(loss)*n; n_tokens+=n
    return total/max(n_tokens,1)


def main():
    rows=read_rows()
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True).to('cuda:0')
    model=PeftModel.from_pretrained(base,ADAPTER,is_trainable=False,local_files_only=True).eval()
    im_end=tok.convert_tokens_to_ids('<|im_end|>')
    stop_ids=[tok.eos_token_id]+([im_end] if im_end!=tok.eos_token_id else [])
    result_path=ROOT/'reports/adapter_scale_comparison.jsonl'
    report_path=ROOT/'reports/adapter_scale_comparison.md'
    all_results=[]; started=time.time()
    for scale in SCALES:
        nll=validation_nll(model,rows,tok,scale)
        with rescale_adapter_scale(model,scale),torch.inference_mode():
            for i,row in enumerate(rows):
                inputs=encode_prompt(row,tok).to('cuda:0')
                output=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,
                    repetition_penalty=1.1,no_repeat_ngram_size=3,eos_token_id=stop_ids,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
                new=output[0,inputs['input_ids'].shape[1]:]
                last=int(new[-1]) if len(new) else None
                stop='im_end' if last==im_end else ('eos' if last==tok.eos_token_id else 'max_new_tokens')
                all_results.append({'scale':scale,'sample_id':i,'question':row['messages'][0]['content'],
                    'reference':row['messages'][-1]['content'],'response':tok.decode(new,skip_special_tokens=True).strip(),
                    'generated_tokens':len(new),'stop_reason':stop})
        print(f'scale={scale:.2f} validation_nll={nll:.4f}',flush=True)
        # NLL 单独记录，生成文本明细在 JSONL 中。
        all_results.append({'scale':scale,'metric':'validation_nll','value':nll})
    # 同一尺度的 NLL 指标行与回答行均保留，便于复现和人工检查。
    with result_path.open('w',encoding='utf-8') as f:
        for item in all_results: f.write(json.dumps(item,ensure_ascii=False)+'\n')
    answer_rows=[item for item in all_results if 'response' in item]
    stop_counts={scale:Counter(item['stop_reason'] for item in answer_rows if item['scale']==scale) for scale in SCALES}
    lines=['# LoRA Adapter 强度对照','',
        '- 数据：现有 validation split（30 条）；测试集未读取。',
        '- 对照：Base（scale=0）以及 LoRA scale 0.25/0.5/0.75/1.0。',
        f'- 生成：贪心解码，max_new_tokens={MAX_NEW_TOKENS}，repetition_penalty=1.1，no_repeat_ngram_size=3。',
        '- 每个问题、每个 scale 使用完全相同的 prompt 和解码参数。',
        '- 质量仍需人工查看逐题输出；验证 NLL 不能单独代表回答质量。','',
        '| Adapter scale | Validation NLL |','|---:|---:|']
    for item in all_results:
        if item.get('metric')=='validation_nll': lines.append(f"| {item['scale']:.2f} | {item['value']:.4f} |")
    lines += ['', '## 停止情况', '', '| Scale | im_end | eos | max_new_tokens |', '|---:|---:|---:|---:|']
    for scale in SCALES:
        c=stop_counts[scale]
        lines.append(f"| {scale:.2f} | {c.get('im_end',0)} | {c.get('eos',0)} | {c.get('max_new_tokens',0)} |")
    lines += ['', '- 生成是否自然、正确需要人工查看；NLL 不能单独代表回答质量。', f'- 逐题输入、参考答案、生成结果和停止原因：`{result_path}`',
        f'- 对照耗时：{time.time()-started:.1f} 秒。','']
    report_path.write_text('\n'.join(lines),encoding='utf-8')
    print(f'results={result_path}\nreport={report_path}',flush=True)

if __name__=='__main__': main()
