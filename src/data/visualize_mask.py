"""显示一个已 token 化样本的 token、label 和 loss 参与情况。"""
import argparse, json
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]
MODEL=Path('/data/shuaiz/models/Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--split',default='train',choices=['train','validation','test']); p.add_argument('--index',type=int,default=0); a=p.parse_args()
    src=ROOT/f'data/processed/{a.split}_tokenized.jsonl'
    rows=[json.loads(x) for x in src.read_text(encoding='utf-8').splitlines()]
    row=rows[a.index]; tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    print('split/index:',a.split,a.index,'prompt_length:',row['prompt_length'],'full_length:',row['full_length'],'truncated:',row['truncated'])
    print(f"{'pos':>4} {'id':>7} {'loss':>5} token")
    for i,(tid,label) in enumerate(zip(row['input_ids'],row['labels'])):
        piece=tok.convert_ids_to_tokens(tid)
        print(f'{i:4d} {tid:7d} {"YES" if label != -100 else "no":>5} {piece}')
    print('\nDecoded full text:')
    print(tok.decode(row['input_ids'],skip_special_tokens=False))
    print('\nDecoded supervised target:')
    print(tok.decode([x for x in row['labels'] if x != -100],skip_special_tokens=False))
if __name__=='__main__': main()
