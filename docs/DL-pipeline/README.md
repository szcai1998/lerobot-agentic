# Deep Learning Pipeline Documentation

Welcome to the Deep Learning & Neural Policy training guide for `lerobot-agentic`.

## Contents

- [`training_architecture_and_dataset.md`](./training_architecture_and_dataset.md): Complete technical breakdown of the CVAE Transformer architecture, Action Chunking formulations, LeRobotDataset v2.0 episodic storage format, loss functions, and RTX 3070 training hyperparameters.

## Quick Reference: Training Flow

```bash
# 1. Synthesize 50 expert demonstrations in MuJoCo
python scripts/record_dataset.py --episodes 50 --output-dir data/lerobot_embodied_arm

# 2. Train the ACT Policy on your local RTX 3070 GPU
python scripts/train_policy.py --dataset-dir data/lerobot_embodied_arm --steps 50000 --batch-size 16

# 3. Evaluate the trained policy closed-loop with Gemini Cognitive Supervisor
python scripts/evaluate.py --policy-path outputs/checkpoints/act_embodied_arm/best --episodes 10 --video
```
