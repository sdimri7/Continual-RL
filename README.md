# Diffusion Policy on Push-T

Implementation of **Diffusion Policy** trained on the **Push-T** robotic manipulation task using the **ManiSkill** simulator.

---

## What are we trying to do?

The goal is to teach a robot arm to **push a T-shaped block to a target position** on a table — purely by learning from expert demonstrations, without hand-crafted reward shaping or manual programming of motion.

This is called **Imitation Learning** (or Learning from Demonstrations). The robot watches recorded expert trajectories and learns a policy that can reproduce the same kind of behaviour in new situations.

The specific challenge with robotic manipulation is that the action space is **continuous and multimodal** — at any given moment, there may be several equally valid ways to push the block toward the goal. A policy that just averages over all valid actions will produce a bad action that achieves neither. Diffusion Policy is designed specifically to handle this.

---

## How are we solving it?

### The core idea: treat action generation as denoising

Instead of directly predicting the next action, we frame it as a **denoising problem** borrowed from image generation (DDPM / Stable Diffusion):

1. During **training**, we take a correct action sequence from a demonstration, add random Gaussian noise to it, and ask a neural network to predict what noise was added. This is repeated for many different noise levels.
2. During **inference**, we start from **pure random noise** and run the network repeatedly to gradually remove the noise, step by step, until a clean, executable action sequence emerges.

Because the denoising process can converge to different valid actions from different random starting points, the policy naturally captures **multiple modes** of behaviour — exactly what averaging-based policies cannot do.

### Observation conditioning

The denoiser does not work blindly. At every denoising step it is shown:

- A short **history of camera images** (the last 2 frames) so it knows the current state of the scene.
- The current **diffusion timestep** so it knows how noisy its input is and how aggressively to denoise.

Both pieces of information are injected into the network through **FiLM conditioning** (Feature-wise Linear Modulation), which scales and shifts internal feature maps at every layer based on the conditioning signal.

### Predict a horizon, execute a slice

Rather than predicting one action at a time, the policy predicts a **window of 16 future actions** at once. Of those 16, only the first 8 are actually executed before the policy is called again. This receding-horizon approach gives the policy enough lookahead to plan smooth motions while staying reactive to changes in the scene.

---

## Implementation 

### Diffusion Policy
Refernce from official implementation at https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/diffusion_policy has been taken 

## Setup

Create env : 

```
cd official_diffusion policy
conda create -n diffusion-policy-ms python=3.9
conda activate diffusion-policy-ms
pip install -e .
```

Download raw demo:
```
python -m mani_skill.utils.download_demo "PickCube-v1"
```

Replay preprocess dataset:
```
DEMO_PATH=~/.maniskill/demos

PickCube:
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ${DEMO_PATH}/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state \
  -c pd_ee_delta_pos \
  -o state \
  --save-traj \
  --num-envs 10 \
  -b physx_cpu
  ```


