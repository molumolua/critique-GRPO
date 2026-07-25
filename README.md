# **[ICML 2026 Spotlight] Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback**  

[![Paper](https://img.shields.io/badge/arXiv-2506.03106-b31b1b.svg)](https://arxiv.org/abs/2506.03106)
[![Model](https://img.shields.io/badge/🤗%20Model-Critique_GRPO_Qwen3--8B-blue)](https://huggingface.co/xyingzhang/critique_grpo_math_4k_qwen3_8b_rollout7_self_critique_1_global_step_300)

![Method Overview](Introduction.png)

## Overview

Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language model (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. 

---

#### Critique-GRPO Framework
![Critique-GRPO Framework](Critique_GRPO.png)

---

## 🔥🔥🔥 Installation & Training (Update the training code)

### Build Training Environment

```bash
# If using conda (recommended):
conda env create -f training_env.yml
conda activate critique-grpo
# Make the script executable
chmod +x verl/examples/grpo_trainer/run_open_r1_math4k-qwen3-8b-base-critique_simple_gt_online.sh

# Execute the training script
bash verl/examples/grpo_trainer/run_open_r1_math4k-qwen3-8b-base-critique_simple_gt_online.sh
```

### DenoiseRL-v2-aligned offline baselines

Two recipes align Critique-GRPO with the DenoiseRL-v2 4B and 8B runs. They use
the same local model family, train/evaluation parquet files, batch size,
16-rollout group size, 8192/4096 prompt/response limits, sampling parameters,
token-mean loss, PPO clipping, optimizer schedule, validation cadence, and
four-GPU layout. The training pool also applies DenoiseRL-v2's ordered
`wrong_answer_with_boxed` usability filter (6332 of 7500 local rows). Hugging
Face and W&B are offline by default.

```bash
bash verl/examples/grpo_trainer/run_denoise_v2-qwen3-4b-base-critique_grpo.sh
bash verl/examples/grpo_trainer/run_denoise_v2-qwen3-8b-base-critique_grpo.sh
```

The recipes expect `DenoiseRL` to be a sibling directory by default. Set
`DENOISE_ROOT`, `MODEL_PATH`, `TRAIN_FILE`, or the `VAL_*` variables to other
local paths when needed. `CRITIQUE_TYPE=simple_gt` performs critique generation
locally and never calls the optional Azure/OpenAI path.

Please email Xiaoying at zhangxycuhk@gmail.com with any questions. (Note that CUHK-related email addresses are no longer functional.)

## Ackowledgement
Our code builds upon several excellent open-source projects: VERL (https://github.com/volcengine/verl), LUFFY (https://github.com/ElliottYan/LUFFY).
We extend our gratitude to the team members and the broader research community for their contributions.


## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2025critique,
  title={Critique-grpo: Advancing llm reasoning with natural language and numerical feedback},
  author={Zhang, Xiaoying and Zhang, Yipeng and Sun, Hao and Feng, Kaituo and Lu, Chaochao and Yang, Chao and Meng, Helen},
  journal={arXiv preprint arXiv:2506.03106},
  year={2025}
}
