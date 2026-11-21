import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = (
    "/data/shuaiz/models/"
    "Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda:0")

messages = [
    {"role": "user", "content": "八加九等于多少"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda:0")

with torch.inference_mode():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
    )

# 只解码新生成的回答，不重复打印输入
answer_ids = output_ids[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(answer_ids, skip_special_tokens=True))