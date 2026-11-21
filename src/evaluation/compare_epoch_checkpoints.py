"""对同一次训练的 epoch 1/3/6 adapter 快照做固定提示生成比较。"""
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/data/shuaiz/models/Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987')
RUN=ROOT/'outputs/lora/epoch_sweep_6epoch/checkpoints'
CHECKPOINTS={1:RUN/'epoch_01',3:RUN/'epoch_03',6:RUN/'epoch_06'}
MAX_NEW=96
PROMPTS=[
 {'name':'validation_overfitting','messages':[{'role':'user','content':'什么是过拟合？'}],
  'reference':'过拟合是模型过度贴合训练样本中的细节或噪声，导致训练表现好、未见数据表现差。可用验证集监测，并通过正则化、早停或增加多样数据缓解。'},
 {'name':'external_general_probe','messages':[{'role':'user','content':'后训练是什么？'}], 'reference':None},
]
def main():
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 eot=tok.eos_token_id; im_end=tok.convert_tokens_to_ids('<|im_end|>'); stop=[eot]+([im_end] if im_end!=eot else [])
 results=[]
 for epoch,adapter in CHECKPOINTS.items():
  base=AutoModelForCausalLM.from_pretrained(BASE,dtype=torch.bfloat16,local_files_only=True).to('cuda:0')
  model=PeftModel.from_pretrained(base,adapter,is_trainable=False,local_files_only=True).eval()
  for prompt in PROMPTS:
   inputs=tok.apply_chat_template(prompt['messages'],add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda:0')
   with torch.inference_mode():
    out=model.generate(**inputs,max_new_tokens=MAX_NEW,do_sample=False,repetition_penalty=1.1,no_repeat_ngram_size=3,eos_token_id=stop,pad_token_id=tok.pad_token_id or eot)
   new=out[0,inputs['input_ids'].shape[1]:]
   last=int(new[-1]) if len(new) else None
   stop_reason='im_end' if last==im_end else ('eos' if last==eot else 'max_new_tokens')
   results.append({'epoch':epoch,'checkpoint':str(adapter),'prompt_name':prompt['name'],'question':prompt['messages'][0]['content'],'reference':prompt['reference'],'response':tok.decode(new,skip_special_tokens=True).strip(),'generated_tokens':len(new),'stop_reason':stop_reason})
  del model,base; torch.cuda.empty_cache()
  print(f'completed epoch={epoch}',flush=True)
 out=ROOT/'reports/epoch_generation_comparison.jsonl'
 with out.open('w',encoding='utf-8') as f:
  for row in results:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 lines=['# Epoch Adapter 生成对照','','- Checkpoints：同一条 6-epoch 训练轨迹的 epoch 1、3、6。','- 解码：greedy，max_new_tokens=96，repetition_penalty=1.1，no_repeat_ngram_size=3。','- 只使用 validation 题和一条额外通用探针；未读取 test split。','']
 for p in PROMPTS:
  lines += [f"## {p['name']}",f"\n问题：{p['messages'][0]['content']}\n"]
  if p['reference']: lines += [f"验证集参考答案：{p['reference']}\n"]
  for row in results:
   if row['prompt_name']==p['name']: lines += [f"### Epoch {row['epoch']}（停止：{row['stop_reason']}，{row['generated_tokens']} tokens）",'',row['response'],'']
 lines += [f'- 完整 JSONL：`{out}`','']
 (ROOT/'reports/epoch_generation_comparison.md').write_text('\n'.join(lines),encoding='utf-8')
 print(f'results={out}',flush=True)
if __name__=='__main__':main()
