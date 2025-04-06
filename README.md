# Minimum Implementation of GRPO

[Lab 1](https://github.com/GAIR-NLP/cs2916/tree/main/2025/hw1) of SJTU CS2916 Large Language Models.

Based on the GRPO implementation in the original repository, this repo made several key modifications to improve the performance of the GRPO algorithm. The main changes include:
1. **Clip Higher**: Increase the clip ratio of positive advantage actions from $0.2$ to $0.28$ to reward correct answers more.
2. **Remove the KL Penalty**: The KL penalty is removed to allow for more exploration.
3. **Modified Learning Rate Schedule**: Adjusted the cosine learning rate scheduler to decay from 1.0*lr to 0.8*lr (instead of the original 0.1*lr) to maintain higher learning rates in later training stages, addressing the observed slower improvement in the second half of training.

The first two modifications are based on the findings from the paper "[DAPO](https://arxiv.org/abs/2503.14476): An Open-Source LLM Reinforcement Learning System at Scale".

## Quickstart

#### Environment setup
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

#### SFT Training (Same as the original repo, for reference only)
```bash
export WANDB_API_KEY=your_wandb_api_key
./scripts/sft_cot.sh
```
Expected training time: ~3 minutes on 4 H800 GPUs.

#### GRPO(Improved) Training
```bash
export WANDB_API_KEY=your_wandb_api_key
export PYTHONPATH=$(pwd)${PYTHONPATH:+:$PYTHONPATH}
./scripts/grpo.sh
```
Expected training time: ~14 hours on 4 H800 GPUs (Most notable improvement is in the first 2 hours).

#### Evaluation
```bash
./scripts/eval.sh ../ckpt/path/to/model
```

Refer to the [original repo](https://github.com/GAIR-NLP/cs2916/tree/main/2025/hw1) for the baseline GRPO implementation as some modifications are directly hardcoded rather than configurable.

## Performance

Base model: [Qwen 2.5 Math 1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B).

| Model          | GSM8k | MATH | AMC23 | OlympiadBench |
|----------------|-------|------|-------|---------------|
| SFT            | 71.6  | 44   | 30    | 9.6           |
| GRPO (baseline)| 74.1  | 61.3 | 50    | 17.4          |
| GRPO (ours)    | 84.4  | 64.6 | 52.5  | 28.4          |