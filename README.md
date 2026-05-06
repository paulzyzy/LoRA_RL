# Low-Rank Adaptation for Critic Learning in Off-Policy Reinforcement Learning

This repository contains the code for the paper:

**Low-Rank Adaptation for Critic Learning in Off-Policy Reinforcement Learning**  
arXiv: https://arxiv.org/abs/2604.18978

The implementation applies LoRA to critic residual-block weights in three off-policy RL settings. Each method folder contains one example script for the main LoRA configuration.

## Repository Structure

- `SAC_BRC`: LoRA critic implementation on top of Bigger Regularized Categorical SAC.
- `FastTD3_SimbaV2`: LoRA critic implementation on top of FastTD3 with SimbaV2 networks.
- `SAC_SimbaV2`: LoRA critic implementation on top of SAC with SimbaV2 networks.
- `BiggerRegularizedCategorical`, `FastTD3`, `SimbaV2`: original baseline snapshots kept for reference.

The original upstream repositories are:

- Bigger Regularized Categorical: https://github.com/naumix/BiggerRegularizedCategorical
- FastTD3: https://github.com/younggyoseo/FastTD3
- SimbaV2: https://github.com/DAVIAN-Robotics/SimbaV2


## Environments

Use one environment per setting. CUDA, JAX, and PyTorch versions are sensitive to your driver and machine; the commands below follow the checked-in requirements.

### FastTD3 + SimbaV2 + LoRA

```bash
cd FastTD3_SimbaV2
conda create -n fasttd3_lora python=3.10 -y
conda activate fasttd3_lora
pip install -r requirements/requirements_playground.txt
pip install -e .
```

Example run:

```bash
CONDA_ENV=fasttd3_lora GPU=0 ENV_NAME=G1JoystickFlatTerrain SEED=0 WANDB_MODE=offline \
  bash scripts/run_lora_example.sh
```

### SAC-BRC + LoRA

```bash
cd SAC_BRC
conda create -n brc_lora python=3.10 -y
conda activate brc_lora
pip install -r requirements.txt
```

Example run:

```bash
CONDA_ENV=brc_lora GPU=0 ENV_NAMES=humanoid-walk SEED=0 WANDB_MODE=offline \
  bash scripts/run_lora_example.sh
```

### SAC + SimbaV2 + LoRA

```bash
cd SAC_SimbaV2
conda create -n simba_lora python=3.10 -y
conda activate simba_lora
pip install -r deps/requirements.txt
pip install -e .
```

Example run:

```bash
CONDA_ENV=simba_lora GPU=0 ENV_NAME=humanoid-run SEED=0 WANDB_MODE=offline \
  bash scripts/run_lora_example.sh
```

## Citation

```bibtex
@misc{zhuang2026lowrankadaptationcriticlearning,
      title={Low-Rank Adaptation for Critic Learning in Off-Policy Reinforcement Learning}, 
      author={Yuan Zhuang and Yuexin Bian and Sihong He and Jie Feng and Qing Su and Songyang Han and Jonathan Petit and Shihao Ji and Yuanyuan Shi and Fei Miao},
      year={2026},
      eprint={2604.18978},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.18978}, 
}
```
