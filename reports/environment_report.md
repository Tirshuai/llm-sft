# Environment Check Report

Date: 2026-11-20 (server local time, Asia/Shanghai)

## Host

- OS: Ubuntu 22.04.3 LTS (Jammy), x86_64; kernel 5.15.0-191-generic.
- User/home: `shuaiz`, `/home/shuaiz`.
- CPU: 224 logical processors.
- RAM: 1.0 TiB total, about 990 GiB available during inspection; swap 8 GiB.
- Root filesystem: 879 GiB total, 238 GiB available at inspection.
- Data filesystem mounted at `/data`: 7.0 TiB total, about 1.2 TiB available. `/data/shuaiz` existed and occupied about 522 GiB.

## Python and Conda

- System Python: 3.10.12.
- Miniconda: `/home/shuaiz/miniconda3`, version 26.5.3.
- Conda package cache: `/data/shuaiz/conda_pkgs`.
- Project environment: `/data/shuaiz/conda_envs/llm_sft`, Python 3.11.18.
- Environment creation used the configured Tsinghua Anaconda mirror.
- The new environment is isolated from existing environments. Non-interactive SSH shells do not automatically put Conda on PATH; use `source /home/shuaiz/miniconda3/etc/profile.d/conda.sh && conda activate llm_sft`, or invoke `/data/shuaiz/conda_envs/llm_sft/bin/python` directly.

## GPU and CUDA

- NVIDIA driver: 580.178.04; `nvidia-smi` reports CUDA 13.0 compatibility.
- GPUs: 10 x NVIDIA GeForce RTX 4090 D, compute capability 8.9, 49,140 MiB per GPU.
- All 10 devices were visible to PyTorch and were nearly idle during inspection (about 1 MiB used per GPU).
- PyTorch in `llm_sft`: 2.8.0+cu128, CUDA runtime build 12.8.
- Verified `torch.cuda.is_available() == True`, 10 devices visible, and `torch.cuda.is_bf16_supported() == True`.
- A BF16 tensor operation succeeded on `cuda:0`.
- bitsandbytes 0.45.5 `Linear4bit` with NF4 performed a small BF16 forward pass on `cuda:0` successfully. This verifies the 4-bit kernel path, not loading or training a quantized language model.

## Installed ML stack in `llm_sft`

- PyTorch 2.8.0+cu128
- Transformers 4.57.1
- PEFT 0.20.0
- TRL 0.25.0
- Accelerate 1.10.1
- bitsandbytes 0.45.5
- datasets 5.0.1
- NumPy 2.2.6
- PyYAML 6.0.3, SentencePiece 0.2.3, protobuf 7.36.2, safetensors 0.8.0

Imports for the listed framework libraries succeeded. The pinned local Qwen2.5-0.5B loaded successfully in BF16 on `cuda:0`. Its `Qwen2TokenizerFast` tokenizer exposes a chat template; a template encoding, forward pass, and 24-token greedy generation succeeded. This was an inference smoke check, not a training run.

## Network and model cache

- `huggingface.co` HTTPS timed out during the connectivity check.
- `hf-mirror.com` returned HTTP 200.
- Tsinghua PyPI mirror and Tsinghua Anaconda mirror were reachable; PyTorch and Python dependencies were installed through these mirrors.
- Existing Hugging Face cache under `/home/shuaiz/.cache/huggingface` was about 164 KiB and contained a Uni-NaVid repository directory; it did not contain a ready cache for the planned teaching model.
- The pinned Qwen2.5-0.5B model (revision `060db6499f32faf8b98477b0a26969ef7d8b9987`) was downloaded into `/data/shuaiz/models/Qwen2.5-0.5B-060db6499f32faf8b98477b0a26969ef7d8b9987`. The HF mirror weight transfer stalled; the identical `model.safetensors` artifact was resumed from the Qwen ModelScope repository, whose linked ETag matched the HF artifact SHA-256 (`88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342`). A 2 MiB range probe measured about 3.1 MB/s from the server through ModelScope; resumed transfer sustained about 9.3 MB/s. Direct Hugging Face access timed out from the server and this Windows client. Use the pinned local model path for experiments.

## Current project scope and limitations

- First training baseline will use one GPU; multi-GPU training is not enabled.
- CUDA, BF16, and a bitsandbytes NF4 layer were tested. Full QLoRA model loading/training, assistant-only loss masking, data processing, and training are not yet tested.
- No system CUDA, existing Conda environment, or pre-existing project directory was modified.
