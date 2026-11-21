"""在保留的 test split 上统一比较 Base、LoRA、QLoRA。"""
import json
from collections import Counter
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data/shuaiz/models/Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987')
LORA=ROOT/'outputs/lora/experiment_001'
QLORA=ROOT/'outputs/qlora/experiment_001'
MAX_NEW_TOKENS=96

def load_model(name):
    if name=='Base':
        return AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True).to('cuda:0')
    if name=='LoRA':
        base=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True).to('cuda:0')
        return PeftModel.from_pretrained(base,LORA,is_trainable=False,local_files_only=True)
    if name=='QLoRA':
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        base=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,quantization_config=q,device_map={'':0},local_files_only=True)
        if not getattr(base,'is_loaded_in_4bit',False): raise RuntimeError('QLoRA evaluation base is not actually 4-bit')
        return PeftModel.from_pretrained(base,QLORA,is_trainable=False,local_files_only=True)
    raise ValueError(name)

def main():
    rows=[json.loads(s) for s in (ROOT/'data/test.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    im_end=tok.convert_tokens_to_ids('<|im_end|>')
    stop_ids=[tok.eos_token_id]+([im_end] if im_end!=tok.eos_token_id else [])
    results=[]
    for name in ('Base','LoRA','QLoRA'):
        model=load_model(name).eval()
        if name=='QLoRA':
            layers=sum(m.__class__.__name__=='Linear4bit' for m in model.modules())
            if layers==0: raise RuntimeError('No Linear4bit layers found for QLoRA inference')
            print(f'QLoRA inference Linear4bit layers={layers}',flush=True)
        for i,row in enumerate(rows):
            messages=row['messages'][:-1]  # 标准答案不会进入输入
            inputs=tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda:0')
            with torch.inference_mode():
                out=model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,repetition_penalty=1.1,no_repeat_ngram_size=3,eos_token_id=stop_ids,pad_token_id=tok.pad_token_id or tok.eos_token_id)
            new=out[0,inputs['input_ids'].shape[1]:]
            last=int(new[-1]) if len(new) else None
            reason='im_end' if last==im_end else ('eos' if last==tok.eos_token_id else 'max_new_tokens')
            results.append({'model':name,'sample_id':i,'question':messages[-1]['content'],'reference':row['messages'][-1]['content'],'response':tok.decode(new,skip_special_tokens=True).strip(),'generated_tokens':len(new),'stop_reason':reason})
        del model; torch.cuda.empty_cache()
        print(f'completed {name}',flush=True)
    jsonl=ROOT/'reports/generation_comparison.jsonl'
    with jsonl.open('w',encoding='utf-8') as f:
        for r in results:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    lines=['# Base / LoRA / QLoRA Test 对比','','- 数据：保留的 test split，共 30 条；每次只向模型输入 user 消息，没有把参考答案放入 prompt。',
      f'- 统一生成参数：greedy，max_new_tokens={MAX_NEW_TOKENS}，repetition_penalty=1.1，no_repeat_ngram_size=3；EOS 同时监听 `<|endoftext|>` 与 `<|im_end|>`。',
      '- LoRA 用 BF16 Base + LoRA adapter；QLoRA 用 4-bit NF4 Base + QLoRA adapter。',
      '- 生成质量需人工判断；token loss 或关键词规则不能替代完整能力评估。','',
      '| Model | samples | im_end stops | eos stops | max token stops |','|---|---:|---:|---:|---:|']
    for name in ('Base','LoRA','QLoRA'):
        subset=[r for r in results if r['model']==name]; c=Counter(r['stop_reason'] for r in subset)
        lines.append(f"| {name} | {len(subset)} | {c.get('im_end',0)} | {c.get('eos',0)} | {c.get('max_new_tokens',0)} |")
    lines += ['',f'- 完整问题、参考答案、生成内容和停止原因：`{jsonl}`','']
    (ROOT/'reports/evaluation_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'results={jsonl}\nreport={ROOT/"reports/evaluation_report.md"}',flush=True)
if __name__=='__main__':main()
